from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from dexterous_hand.config import (
    DomainRandomization,
    MjxPegTrainConfig,
    PegRewardConfig,
    PegSceneConfig,
)
from dexterous_hand.envs.gpu.mjx_vec_env import MjxVecEnv
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.scene_builder import GRIP_BIAS, apply_flexion_bias, build_grip_ctrl
from dexterous_hand.rewards.gpu.peg_reward import (
    PegRewardState,
    init_peg_reward_state,
    peg_reward,
)
from dexterous_hand.utils.gpu.mjx_helpers import (
    get_body_axis_jax,
    get_finger_touch_from_sensors,
    get_fingertip_positions_jax,
    get_insertion_depth_jax,
    get_object_state_jax,
    get_palm_position_jax,
    get_peg_hole_relative_jax,
)


class PegEnvState(NamedTuple):
    reward_state: PegRewardState
    previous_actions: jnp.ndarray
    smoothed_actions: jnp.ndarray
    stage: jnp.ndarray
    no_contact_grace: jnp.ndarray
    initial_peg_height: jnp.ndarray
    step_count: jnp.ndarray
    key: jax.Array


class ShadowHandPegMjxEnv(MjxVecEnv):
    def __init__(
        self,
        num_envs: int = 2048,
        seed: int = 42,
        scene_config: PegSceneConfig | None = None,
        reward_config: PegRewardConfig | None = None,
        max_episode_steps: int = 500,
        obs_noise_std: float = 0.0,
        dr: DomainRandomization | None = None,
    ) -> None:
        self.scene_config = scene_config or PegSceneConfig()
        self.reward_config = reward_config or PegRewardConfig()
        if self.scene_config.spawn_max_radius <= self.scene_config.spawn_min_radius:
            raise ValueError(
                f"spawn_max_radius ({self.scene_config.spawn_max_radius}) must exceed "
                f"spawn_min_radius ({self.scene_config.spawn_min_radius})"
            )
        self._episode_limit = max_episode_steps
        self._reward_weights = self.reward_config.weights

        self._p_pre_grasped = jnp.array(0.0)

        super().__init__(num_envs=num_envs, seed=seed, obs_noise_std=obs_noise_std, dr=dr)

        self._rebuild_peg_caches()

    def _build_model(self) -> mujoco.MjModel:
        model, _, _ = build_peg_scene(self.scene_config)
        return model

    def _obs_size(self) -> int:
        return 134

    def _action_size(self) -> int:
        return int(self._cpu_model.nu)

    @property
    def _max_episode_steps(self) -> int:
        return self._episode_limit

    def _rebuild_peg_caches(self) -> None:
        _, _, self._nm = build_peg_scene(self.scene_config)

        init_qpos_table = self._cpu_data.qpos.copy()
        apply_flexion_bias(init_qpos_table, self._cpu_model)
        self._init_qpos_table = jnp.array(init_qpos_table)

        init_qpos_grip = self._cpu_data.qpos.copy()
        apply_flexion_bias(init_qpos_grip, self._cpu_model, bias_map=GRIP_BIAS)
        self._init_qpos_grip = jnp.array(init_qpos_grip)

        self._grip_ctrl = jnp.array(build_grip_ctrl(self._cpu_model))

        self._grasp_site_id = mujoco.mj_name2id(
            self._cpu_model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )

        self._finger_touch_adr = jnp.asarray(
            self._nm.sensor_map.finger_touch_adr, dtype=jnp.int32
        )
        self._wall_force_adr = jnp.asarray(
            self._nm.sensor_map.wall_force_adr, dtype=jnp.int32
        )
        self._fingertip_site_ids = jnp.asarray(self._nm.fingertip_site_ids, dtype=jnp.int32)
        self._peg_length = (
            self.scene_config.peg_half_length * 2.0 + self.scene_config.peg_radius * 2.0
        )

    def set_curriculum_params(self, clearance: float, p_pre_grasped: float) -> None:
        self._p_pre_grasped = jnp.array(float(p_pre_grasped))

        clearance_f = float(clearance)
        clearance_changed = abs(clearance_f - float(self.scene_config.clearance)) > 1e-9

        if clearance_changed:
            self.scene_config.clearance = clearance_f
            self._cpu_model = self._build_model()
            self._cpu_data = mujoco.MjData(self._cpu_model)
            self._mjx_model = mjx.put_model(self._cpu_model)

            n_act = self._action_size()
            self._ctrl_low = jnp.array(self._cpu_model.actuator_ctrlrange[:n_act, 0])
            self._ctrl_high = jnp.array(self._cpu_model.actuator_ctrlrange[:n_act, 1])

            self._rebuild_peg_caches()
            # obs shape depends on model; only re-jit obs when the model rebuilds
            self._batched_get_obs = jax.jit(jax.vmap(self._get_obs_single, in_axes=(None, 0, 0)))

        # re-jit every curriculum change: reset + step close over _p_pre_grasped.
        self._batched_reset = jax.jit(jax.vmap(self._reset_single, in_axes=(None, 0, 0)))
        self._batched_step = self._build_batched_step()

        if clearance_changed and self._mjx_data_batch is not None:
            self.reset()

    def _reset_single(
        self, mjx_model: Any, mjx_data: Any, key: jax.Array
    ) -> tuple[Any, PegEnvState]:
        nm = self._nm
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        spawn_pre_grasped = jax.random.uniform(k4) < self._p_pre_grasped

        qpos = jnp.where(spawn_pre_grasped, self._init_qpos_grip, self._init_qpos_table)

        hand_qpos = qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        noise = jax.random.uniform(k1, shape=hand_qpos.shape, minval=-0.05, maxval=0.05)
        qpos = qpos.at[nm.hand_qpos_start : nm.hand_qpos_end].set(hand_qpos + noise)
        qvel = jnp.zeros(mjx_model.nv)

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.forward(mjx_model, mjx_data)

        # peg spawn: radial sampling around the hole (mirrors CPU peg env)
        min_r = float(self.scene_config.spawn_min_radius)
        max_r = float(self.scene_config.spawn_max_radius)
        r = jax.random.uniform(k2, minval=min_r, maxval=max_r)
        theta = jax.random.uniform(k3, minval=0.0, maxval=2.0 * math.pi)
        hole_x = float(self.scene_config.hole_offset[0])
        hole_y = float(self.scene_config.hole_offset[1])
        table_peg_x = hole_x + r * jnp.cos(theta)
        table_peg_y = hole_y + r * jnp.sin(theta)
        table_peg_z = (
            self.scene_config.table_height
            + self.scene_config.peg_half_length
            + self.scene_config.peg_radius
            + 0.001
        )
        table_xyz = jnp.array([table_peg_x, table_peg_y, table_peg_z])

        pregrasp_xyz = mjx_data.site_xpos[self._grasp_site_id]

        peg_xyz = jnp.where(spawn_pre_grasped, pregrasp_xyz, table_xyz)

        s = nm.peg_qpos_start
        qpos = mjx_data.qpos.at[s : s + 3].set(peg_xyz)
        qpos = qpos.at[s + 3 : s + 7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))

        mjx_data = mjx_data.replace(qpos=qpos, qvel=jnp.zeros(mjx_model.nv))
        mjx_data = mjx.forward(mjx_model, mjx_data)

        # GRIP_BIAS ctrl during settle: ctrl=0 would drive flexion joints to
        # angle 0 (fully open) and drop the peg before the policy ever acts.
        mjx_data = mjx_data.replace(ctrl=self._grip_ctrl)

        def _settle(data: Any, _: Any) -> tuple[Any, None]:
            return mjx.step(mjx_model, data), None

        mjx_data, _ = jax.lax.scan(_settle, mjx_data, None, length=5)

        initial_peg_height = mjx_data.xpos[nm.peg_body_id][2]

        n_act = self._action_size()
        env_state = PegEnvState(
            reward_state=init_peg_reward_state(initial_peg_height),
            previous_actions=jnp.zeros(n_act),
            smoothed_actions=jnp.zeros(n_act),
            stage=jnp.array(0, dtype=jnp.int32),
            no_contact_grace=jnp.array(0, dtype=jnp.int32),
            initial_peg_height=initial_peg_height,
            step_count=jnp.array(0, dtype=jnp.int32),
            key=k5,
        )

        return mjx_data, env_state

    def _step_single(
        self,
        mjx_model: Any,
        mjx_data: Any,
        env_state: PegEnvState,
        action: jax.Array,
    ) -> tuple[Any, PegEnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
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
        finger_pos = get_fingertip_positions_jax(mjx_data.site_xpos, self._fingertip_site_ids)
        peg_pos, peg_quat, peg_linvel, peg_angvel = get_object_state_jax(
            mjx_data.qpos,
            mjx_data.qvel,
            mjx_data.xpos,
            nm.peg_body_id,
            nm.peg_qpos_start,
            nm.peg_qvel_start,
        )

        touch_vals, contact_mask = get_finger_touch_from_sensors(
            mjx_data.sensordata, self._finger_touch_adr
        )
        n_contacts = jnp.sum(contact_mask).astype(jnp.float32)

        peg_axis = get_body_axis_jax(mjx_data.xmat, nm.peg_body_id)
        hole_axis = get_body_axis_jax(mjx_data.xmat, nm.hole_body_id)
        hole_pos = mjx_data.xpos[nm.hole_body_id]

        peg_half_length = self.scene_config.peg_half_length
        peg_radius = self.scene_config.peg_radius
        insertion_depth = get_insertion_depth_jax(
            mjx_data.xpos,
            mjx_data.xmat,
            nm.peg_body_id,
            nm.hole_body_id,
            peg_half_length,
            peg_radius,
        )

        # per-wall contact force magnitudes (sum over all hole walls)
        wall_vals = mjx_data.sensordata[self._wall_force_adr]
        contact_force_mag = jnp.sum(wall_vals)

        # curriculum stage gating
        fingers_on_peg = n_contacts >= 2
        peg_lifted = peg_pos[2] > env_state.initial_peg_height + 0.02
        peg_near_hole = jnp.linalg.norm(peg_pos[:2] - hole_pos[:2]) < 0.03
        peg_aligned = jnp.abs(jnp.dot(peg_axis, hole_axis)) > 0.95

        new_grace = jnp.where(
            n_contacts == 0, env_state.no_contact_grace + 1, jnp.array(0, dtype=jnp.int32)
        )

        target = jnp.array(0, dtype=jnp.int32)
        target = jnp.where(fingers_on_peg, 1, target)
        target = jnp.where(peg_lifted, 2, target)
        target = jnp.where(peg_near_hole & peg_aligned, 3, target)

        new_stage = jnp.where(new_grace >= 5, 0, jnp.maximum(env_state.stage, target))

        peg_height = peg_pos[2]

        total, new_reward_state, info = peg_reward(
            state=env_state.reward_state,
            stage=new_stage,
            finger_positions=finger_pos,
            peg_position=peg_pos,
            peg_axis=peg_axis,
            hole_position=hole_pos,
            hole_axis=hole_axis,
            insertion_depth=insertion_depth,
            contact_force_magnitude=contact_force_mag,
            finger_contact_mask=contact_mask,
            peg_height=peg_height,
            peg_linvel=peg_linvel,
            actions=smoothed,
            previous_actions=env_state.previous_actions,
            weights=self._reward_weights,
            peg_length=self._peg_length,
            lift_target=self.reward_config.lift_target,
            table_height=self.scene_config.table_height,
            drop_penalty_value=self.reward_config.drop_penalty,
            complete_bonus=self.reward_config.complete_bonus,
            force_threshold=self.reward_config.force_threshold,
            idle_stage0_penalty=self.reward_config.idle_stage0_penalty,
            lateral_gate_k=self.reward_config.lateral_gate_k,
            idle_stage_cutoff=self.reward_config.idle_stage_cutoff,
            success_threshold=self.reward_config.success_threshold,
            peg_hold_steps=self.reward_config.peg_hold_steps,
            reach_tanh_k=self.reward_config.reach_tanh_k,
            fingertip_weights=self.reward_config.fingertip_weights,
            depth_reward_scale=self.reward_config.depth_reward_scale,
            idle_grace_steps=self.reward_config.idle_grace_steps,
        )


        insertion_complete = (
            insertion_depth > self.reward_config.success_threshold * self._peg_length
        ) & (new_reward_state.insertion_hold_steps >= self.reward_config.peg_hold_steps)
        fell = peg_pos[2] < self.scene_config.table_height - 0.1
        done = insertion_complete | fell
        info["is_success"] = insertion_complete.astype(jnp.float32)

        new_env_state = PegEnvState(
            reward_state=new_reward_state,
            previous_actions=smoothed,
            smoothed_actions=smoothed,
            stage=new_stage,
            no_contact_grace=new_grace,
            initial_peg_height=env_state.initial_peg_height,
            step_count=env_state.step_count + 1,
            key=env_state.key,
        )

        obs = self._get_obs_single(mjx_model, mjx_data, new_env_state)

        return mjx_data, new_env_state, obs, total, done, info

    def _get_obs_single(self, mjx_model: Any, mjx_data: Any, env_state: PegEnvState) -> jax.Array:
        nm = self._nm

        joint_pos = mjx_data.qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        joint_vel = mjx_data.qvel[nm.hand_qvel_start : nm.hand_qvel_end]

        peg_pos, peg_quat, peg_linvel, peg_angvel = get_object_state_jax(
            mjx_data.qpos,
            mjx_data.qvel,
            mjx_data.xpos,
            nm.peg_body_id,
            nm.peg_qpos_start,
            nm.peg_qvel_start,
        )

        hole_pos = mjx_data.xpos[nm.hole_body_id]
        hole_quat = mjx_data.xquat[nm.hole_body_id]

        rel_pos, ang_error = get_peg_hole_relative_jax(
            mjx_data.xpos, mjx_data.xmat, nm.peg_body_id, nm.hole_body_id
        )

        finger_pos = get_fingertip_positions_jax(mjx_data.site_xpos, self._fingertip_site_ids)
        fingertip_peg_dist = jnp.linalg.norm(finger_pos - peg_pos, axis=1)

        palm_pos = get_palm_position_jax(mjx_data.xpos, nm.palm_body_id)
        rel_peg_to_palm = peg_pos - palm_pos

        insertion_depth = get_insertion_depth_jax(
            mjx_data.xpos,
            mjx_data.xmat,
            nm.peg_body_id,
            nm.hole_body_id,
            self.scene_config.peg_half_length,
            self.scene_config.peg_radius,
        )

        # per-wall + total contact forces, exposed to the policy
        per_wall_forces = mjx_data.sensordata[self._wall_force_adr]
        contact_force_mag = jnp.sum(per_wall_forces)
        contact_forces = jnp.concatenate([per_wall_forces, jnp.array([contact_force_mag])])

        obs = jnp.concatenate(
            [
                joint_pos,
                joint_vel,
                peg_pos,
                peg_quat,
                peg_linvel,
                peg_angvel,
                hole_pos,
                hole_quat,
                rel_pos,
                ang_error,
                finger_pos.flatten(),
                fingertip_peg_dist,
                rel_peg_to_palm,
                jnp.array([insertion_depth]),
                contact_forces,
                jnp.asarray(env_state.stage, dtype=jnp.float32).reshape(1),
                env_state.previous_actions,
            ]
        )
        assert obs.shape == (self._obs_size(),), (
            f"peg obs shape {obs.shape} != declared ({self._obs_size()},)"
        )
        return obs

    @classmethod
    def from_config(cls, config: MjxPegTrainConfig) -> ShadowHandPegMjxEnv:
        return cls(
            num_envs=config.num_envs,
            seed=config.seed,
            scene_config=config.scene_config,
            reward_config=config.reward_config,
            max_episode_steps=config.max_episode_steps,
            obs_noise_std=config.obs_noise_std,
            dr=config.dr,
        )
