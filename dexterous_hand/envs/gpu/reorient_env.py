from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from dexterous_hand.config import (
    DomainRandomization,
    MjxReorientTrainConfig,
    ReorientRewardConfig,
    ReorientSceneConfig,
)
from dexterous_hand.envs.gpu.mjx_vec_env import MjxVecEnv
from dexterous_hand.envs.reorient_scene_builder import build_reorient_scene
from dexterous_hand.envs.scene_builder import GRIP_BIAS, apply_flexion_bias, build_grip_ctrl
from dexterous_hand.rewards.gpu.reorient_reward import (
    ReorientRewardState,
    init_reorient_reward_state,
    reorient_reward,
)
from dexterous_hand.utils.gpu.mjx_helpers import (
    get_finger_touch_from_sensors,
    get_fingertip_positions_jax,
    get_object_state_jax,
    get_palm_position_jax,
)
from dexterous_hand.utils.gpu.quaternion import (
    quat_conjugate,
    quat_multiply,
    random_quaternion_within_angle,
    sample_target_quat_rel_to_cube,
)


class ReorientEnvState(NamedTuple):
    reward_state: ReorientRewardState
    previous_actions: jnp.ndarray
    smoothed_actions: jnp.ndarray
    target_quat: jnp.ndarray
    max_target_angle: jnp.ndarray
    targets_reached: jnp.ndarray
    palm_z: jnp.ndarray
    step_count: jnp.ndarray
    key: jax.Array


class ShadowHandReorientMjxEnv(MjxVecEnv):
    def __init__(
        self,
        num_envs: int = 2048,
        seed: int = 42,
        scene_config: ReorientSceneConfig | None = None,
        reward_config: ReorientRewardConfig | None = None,
        max_episode_steps: int = 400,
        obs_noise_std: float = 0.0,
        dr: DomainRandomization | None = None,
    ) -> None:
        self.scene_config = scene_config or ReorientSceneConfig()
        self.reward_config = reward_config or ReorientRewardConfig()
        self._episode_limit = max_episode_steps
        self._reward_weights = self.reward_config.weights

        super().__init__(num_envs=num_envs, seed=seed, obs_noise_std=obs_noise_std, dr=dr)

        _, _, self._nm = build_reorient_scene(self.scene_config)
        init_qpos_np = self._cpu_data.qpos.copy()
        apply_flexion_bias(init_qpos_np, self._cpu_model, bias_map=GRIP_BIAS)
        self._init_qpos = jnp.array(init_qpos_np)
        self._grip_ctrl = jnp.array(build_grip_ctrl(self._cpu_model))
        self._finger_touch_adr = jnp.asarray(
            self._nm.sensor_map.finger_touch_adr, dtype=jnp.int32
        )
        self._fingertip_site_ids = jnp.asarray(self._nm.fingertip_site_ids, dtype=jnp.int32)

        self._grasp_site_id = mujoco.mj_name2id(
            self._cpu_model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )

        self._max_target_angle = jnp.array(0.5236)
        self._target_min_angle = float(self.scene_config.target_min_angle)

    def _build_model(self) -> mujoco.MjModel:
        model, _, _ = build_reorient_scene(self.scene_config)
        return model

    def _obs_size(self) -> int:
        return 109

    def _action_size(self) -> int:
        return int(self._cpu_model.nu)

    @property
    def _max_episode_steps(self) -> int:
        return self._episode_limit

    def set_curriculum_stage(self, max_angle: float) -> None:
        self._max_target_angle = jnp.array(float(max_angle))
        self._batched_reset = jax.jit(jax.vmap(self._reset_single, in_axes=(None, 0, 0)))

    def _reset_single(
        self, mjx_model: Any, mjx_data: Any, key: jax.Array
    ) -> tuple[Any, ReorientEnvState]:
        nm = self._nm
        k1, k2, k3, k4 = jax.random.split(key, 4)

        qpos = self._init_qpos

        hand_qpos = qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        noise = jax.random.uniform(k1, shape=hand_qpos.shape, minval=-0.05, maxval=0.05)
        qpos = qpos.at[nm.hand_qpos_start : nm.hand_qpos_end].set(hand_qpos + noise)
        qvel = jnp.zeros(mjx_model.nv)

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.forward(mjx_model, mjx_data)

        palm_pos = get_palm_position_jax(mjx_data.xpos, nm.palm_body_id)
        palm_z = palm_pos[2]

        cube_pos = mjx_data.site_xpos[self._grasp_site_id]
        cube_noise = jax.random.uniform(k2, shape=(3,), minval=-0.008, maxval=0.008)
        cube_pos = cube_pos + cube_noise

        s = nm.cube_qpos_start
        init_quat = random_quaternion_within_angle(k3, 0.3)
        qpos = mjx_data.qpos.at[s : s + 3].set(cube_pos)
        qpos = qpos.at[s + 3 : s + 7].set(init_quat)

        # GRIP_BIAS ctrl during settle: ctrl=0 would drive flexion joints to
        # angle 0 (fully open) and drop the cube before the policy ever acts.
        mjx_data = mjx_data.replace(qpos=qpos, ctrl=self._grip_ctrl)

        def _settle(data: Any, _: Any) -> tuple[Any, None]:
            return mjx.step(mjx_model, data), None

        mjx_data, _ = jax.lax.scan(_settle, mjx_data, None, length=5)

        min_angle_floor = jnp.minimum(
            jnp.asarray(self._target_min_angle), 0.8 * self._max_target_angle
        )
        cube_quat_now = mjx_data.qpos[s + 3 : s + 7]
        target_quat = sample_target_quat_rel_to_cube(
            k4,
            cube_quat_now,
            self._max_target_angle,
            min_angle_rad=min_angle_floor,
        )

        init_cube_pos = mjx_data.xpos[nm.cube_body_id]

        n_act = self._action_size()
        env_state = ReorientEnvState(
            reward_state=init_reorient_reward_state(init_cube_pos),
            previous_actions=jnp.zeros(n_act),
            smoothed_actions=jnp.zeros(n_act),
            target_quat=target_quat,
            max_target_angle=jnp.array(self._max_target_angle),
            targets_reached=jnp.array(0, dtype=jnp.int32),
            palm_z=palm_z,
            step_count=jnp.array(0, dtype=jnp.int32),
            key=key,
        )

        return mjx_data, env_state

    def _step_single(
        self,
        mjx_model: Any,
        mjx_data: Any,
        env_state: ReorientEnvState,
        action: jax.Array,
    ) -> tuple[Any, ReorientEnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        nm = self._nm
        action = jnp.clip(action, -1.0, 1.0)

        alpha = self.scene_config.action_smoothing_alpha
        smoothed = (1.0 - alpha) * env_state.smoothed_actions + alpha * action

        ctrl = self._ctrl_low + (smoothed + 1.0) / 2.0 * (self._ctrl_high - self._ctrl_low)
        mjx_data = mjx_data.replace(ctrl=ctrl)

        def _substep(data: Any, _: Any) -> tuple[Any, None]:
            return mjx.step(mjx_model, data), None

        mjx_data, _ = jax.lax.scan(_substep, mjx_data, None, length=self.scene_config.frame_skip)

        # reward inputs
        fingertip_pos = get_fingertip_positions_jax(mjx_data.site_xpos, self._fingertip_site_ids)
        cube_pos, cube_quat, cube_linvel, cube_angvel = get_object_state_jax(
            mjx_data.qpos,
            mjx_data.qvel,
            mjx_data.xpos,
            nm.cube_body_id,
            nm.cube_qpos_start,
            nm.cube_qvel_start,
        )
        cube_quat = cube_quat / jnp.maximum(jnp.linalg.norm(cube_quat), 1e-6)

        _, contact_mask = get_finger_touch_from_sensors(mjx_data.sensordata, self._finger_touch_adr)

        drop_offset = self.reward_config.drop_height_offset
        threshold_z = env_state.palm_z - drop_offset
        safety = jnp.clip((cube_pos[2] - threshold_z) / drop_offset, 0.0, 1.0)
        drop_factor = 1.0 - (3.0 * safety**2 - 2.0 * safety**3)

        total, new_reward_state, reward_info, target_reached = reorient_reward(
            state=env_state.reward_state,
            cube_quat=cube_quat,
            target_quat=env_state.target_quat,
            cube_pos=cube_pos,
            cube_linvel=cube_linvel,
            finger_positions=fingertip_pos,
            finger_contact_mask=contact_mask,
            actions=smoothed,
            previous_actions=env_state.previous_actions,
            drop_factor=drop_factor,
            weights=self._reward_weights,
            success_threshold=self.reward_config.success_threshold,
            success_hold_steps=self.reward_config.success_hold_steps,
            drop_penalty_value=self.reward_config.drop_penalty,
            contact_bonus_value=self.reward_config.contact_bonus,
            no_contact_penalty_value=self.reward_config.no_contact_penalty,
            min_contacts_for_rotation=self.reward_config.min_contacts_for_rotation,
            angular_progress_clip=self.reward_config.angular_progress_clip,
            tracking_k=self.reward_config.tracking_k,
            orientation_contact_alpha=self.reward_config.orientation_contact_alpha,
        )
        info = {**reward_info, "is_success": target_reached.astype(jnp.float32)}

        new_key, subkey = jax.random.split(env_state.key)
        min_angle_floor = jnp.minimum(
            jnp.asarray(self._target_min_angle), 0.8 * env_state.max_target_angle
        )
        new_target = jax.lax.cond(
            target_reached,
            lambda: sample_target_quat_rel_to_cube(
                subkey,
                cube_quat,
                env_state.max_target_angle,
                min_angle_rad=min_angle_floor,
            ),
            lambda: env_state.target_quat,
        )
        new_targets_reached = jnp.where(
            target_reached,
            env_state.targets_reached + 1,
            env_state.targets_reached,
        )
        # reset reward state when a new target is sampled
        new_reward_state = jax.lax.cond(
            target_reached,
            lambda: init_reorient_reward_state(cube_pos),
            lambda: new_reward_state,
        )

        # Cube drop no longer terminates the episode; the smooth drop_factor
        # already applies per-step penalty + zeroed orientation reward, and
        # ending early let the policy escape the bad-rotation regime.
        done = jnp.array(False)

        new_env_state = ReorientEnvState(
            reward_state=new_reward_state,
            previous_actions=smoothed,
            smoothed_actions=smoothed,
            target_quat=new_target,
            max_target_angle=env_state.max_target_angle,
            targets_reached=new_targets_reached,
            palm_z=env_state.palm_z,
            step_count=env_state.step_count + 1,
            key=new_key,
        )

        obs = self._get_obs_single(mjx_model, mjx_data, new_env_state)

        return mjx_data, new_env_state, obs, total, done, info

    def _get_obs_single(
        self, mjx_model: Any, mjx_data: Any, env_state: ReorientEnvState
    ) -> jax.Array:
        nm = self._nm

        joint_pos = mjx_data.qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        joint_vel = mjx_data.qvel[nm.hand_qvel_start : nm.hand_qvel_end]

        cube_pos, cube_quat, cube_linvel, cube_angvel = get_object_state_jax(
            mjx_data.qpos,
            mjx_data.qvel,
            mjx_data.xpos,
            nm.cube_body_id,
            nm.cube_qpos_start,
            nm.cube_qvel_start,
        )
        cube_quat = cube_quat / jnp.maximum(jnp.linalg.norm(cube_quat), 1e-6)

        fingertip_pos = get_fingertip_positions_jax(mjx_data.site_xpos, self._fingertip_site_ids)
        fingertip_cube_dists = jnp.linalg.norm(fingertip_pos - cube_pos, axis=1)

        err_quat = quat_multiply(quat_conjugate(cube_quat), env_state.target_quat)

        obs = jnp.concatenate(
            [
                joint_pos,
                joint_vel,
                cube_pos,
                cube_quat,
                cube_linvel,
                cube_angvel,
                env_state.target_quat,
                err_quat,
                fingertip_pos.flatten(),
                fingertip_cube_dists,
                env_state.previous_actions,
            ]
        )
        assert obs.shape == (self._obs_size(),), (
            f"reorient obs shape {obs.shape} != declared ({self._obs_size()},)"
        )
        return obs

    @classmethod
    def from_config(cls, config: MjxReorientTrainConfig) -> ShadowHandReorientMjxEnv:
        return cls(
            num_envs=config.num_envs,
            seed=config.seed,
            scene_config=config.scene_config,
            reward_config=config.reward_config,
            max_episode_steps=config.max_episode_steps,
            obs_noise_std=config.obs_noise_std,
            dr=config.dr,
        )
