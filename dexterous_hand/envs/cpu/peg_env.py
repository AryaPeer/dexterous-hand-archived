from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from dexterous_hand.config import PegRewardConfig, PegSceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.scene_builder import GRIP_BIAS, apply_flexion_bias
from dexterous_hand.rewards.cpu.peg_reward import PegRewardCalculator
from dexterous_hand.utils.cpu.mujoco_helpers import (
    get_body_axis,
    get_finger_contacts,
    get_fingertip_positions,
    get_insertion_depth,
    get_object_state,
    get_palm_position,
    get_peg_hole_relative,
)


class ShadowHandPegEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        scene_config: PegSceneConfig | None = None,
        reward_config: PegRewardConfig | None = None,
    ) -> None:
        """Shadow Hand peg-in-hole env (CPU mujoco).

        @param render_mode: 'human' for viewer, 'rgb_array' for offscreen
        @type render_mode: str | None
        @param scene_config: peg scene physics + layout
        @type scene_config: PegSceneConfig | None
        @param reward_config: peg reward weights and thresholds
        @type reward_config: PegRewardConfig | None
        """

        super().__init__()

        self.scene_config = scene_config or PegSceneConfig()
        self.reward_config = reward_config or PegRewardConfig()
        self.render_mode = render_mode

        # build scene and spaces
        self.model, self.data, self.nm = build_peg_scene(self.scene_config)

        n_obs = 134
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nm.n_actuators,), dtype=np.float32
        )

        # reward
        self.reward_calculator = PegRewardCalculator(
            config=self.reward_config,
            table_height=self.scene_config.table_height,
            peg_half_length=self.scene_config.peg_half_length,
            peg_radius=self.scene_config.peg_radius,
        )

        # state tracking
        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._smoothed_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._stage = 0
        self._no_contact_grace = 0
        self._p_pre_grasped = 0.0
        self._clearance = self.scene_config.clearance
        self._initial_peg_height = self.scene_config.table_height

        self._init_qpos_table = self._build_biased_qpos(bias_map=None)
        self._init_qpos_grip = self._build_biased_qpos(bias_map=GRIP_BIAS)
        self._init_qpos = self._init_qpos_table
        self._grasp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )

        # rendering
        self._renderer: mujoco.Renderer | None = None
        if render_mode == "human":
            self._viewer: mujoco.viewer.Handle | None = None

    def _build_biased_qpos(self, bias_map: dict[str, float] | None) -> np.ndarray:
        qpos = self.data.qpos.copy()
        if bias_map is not None:
            apply_flexion_bias(qpos, self.model, bias_map=bias_map)
        return qpos

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset peg-in-hole episode (rebuilds the scene if clearance changed).

        @param seed: random seed
        @type seed: int | None
        @param options: unused
        @type options: dict[str, Any] | None
        @return: (obs (131,), info with current curriculum stage)
        @rtype: tuple[np.ndarray, dict[str, Any]]
        """

        super().reset(seed=seed)

        if self._clearance != self.scene_config.clearance:
            self.scene_config.clearance = self._clearance
            self.model, self.data, self.nm = build_peg_scene(self.scene_config)
            self._init_qpos_table = self._build_biased_qpos(bias_map=None)
            self._init_qpos_grip = self._build_biased_qpos(bias_map=GRIP_BIAS)
            self._init_qpos = self._init_qpos_table
            self._grasp_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
            )

        mujoco.mj_resetData(self.model, self.data)

        spawn_pre_grasped = self.np_random.random() < self._p_pre_grasped
        init_qpos = self._init_qpos_grip if spawn_pre_grasped else self._init_qpos_table

        hand_qpos = init_qpos[self.nm.hand_qpos_start : self.nm.hand_qpos_end]
        noise = self.np_random.uniform(-0.05, 0.05, size=hand_qpos.shape)
        self.data.qpos[self.nm.hand_qpos_start : self.nm.hand_qpos_end] = hand_qpos + noise
        mujoco.mj_forward(self.model, self.data)

        s = self.nm.peg_qpos_start
        if spawn_pre_grasped:
            # match GPU peg env: place peg at the grasp_site so closed fingers are already in contact.
            peg_pos = self.data.site_xpos[self._grasp_site_id].copy()
            self.data.qpos[s : s + 3] = peg_pos
            self.data.qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]
        else:
            # radial sampling to mirror the GPU peg distribution exactly.
            hole_xy = np.array(self.scene_config.hole_offset[:2], dtype=np.float64)
            min_r = self.scene_config.spawn_min_radius
            max_r = 0.05 * float(np.sqrt(2.0))
            r = float(self.np_random.uniform(min_r, max_r))
            theta = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            peg_x = float(hole_xy[0]) + r * float(np.cos(theta))
            peg_y = float(hole_xy[1]) + r * float(np.sin(theta))
            peg_z = (
                self.scene_config.table_height
                + self.scene_config.peg_half_length
                + self.scene_config.peg_radius
                + 0.001
            )
            self.data.qpos[s : s + 3] = [peg_x, peg_y, peg_z]
            self.data.qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._previous_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._smoothed_actions = np.zeros(self.nm.n_actuators, dtype=np.float64)
        self._stage = 0
        self._no_contact_grace = 0
        initial_peg_height = float(self.data.xpos[self.nm.peg_body_id][2])
        self._initial_peg_height = initial_peg_height
        self.reward_calculator.reset(initial_peg_height=initial_peg_height)

        obs = self._get_obs()
        info = {"stage": self._stage}

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)
        alpha = float(np.clip(self.scene_config.action_smoothing_alpha, 0.0, 1.0))
        if alpha > 0.0:
            action = (1.0 - alpha) * self._smoothed_actions + alpha * action
        self._smoothed_actions = action.astype(np.float64).copy()

        low = self.nm.ctrl_ranges[:, 0]
        high = self.nm.ctrl_ranges[:, 1]
        ctrl = low + (action + 1.0) / 2.0 * (high - low)
        self.data.ctrl[: self.nm.n_actuators] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=self.scene_config.frame_skip)

        nm = self.nm

        finger_pos = get_fingertip_positions(self.data, nm.fingertip_site_ids)

        peg_pos, peg_quat, peg_linvel, peg_angvel = get_object_state(
            self.data,
            nm.peg_body_id,
            nm.peg_qpos_start,
            nm.peg_qvel_start,
        )

        num_contacts, contact_finger_indices = get_finger_contacts(
            self.model,
            self.data,
            nm.finger_geom_ids_per_finger,
            nm.peg_geom_id,
        )

        peg_axis = get_body_axis(self.data, nm.peg_body_id)
        hole_axis = get_body_axis(self.data, nm.hole_body_id)
        hole_pos = self.data.xpos[nm.hole_body_id].copy()

        # peg geometry for insertion depth
        peg_half_length = self.scene_config.peg_half_length
        peg_radius = self.scene_config.peg_radius

        insertion_depth = get_insertion_depth(
            self.data, nm.peg_body_id, nm.hole_body_id, peg_half_length, peg_radius
        )

        wall_force_adr = nm.sensor_map.wall_force_adr
        per_wall_forces = np.asarray(self.data.sensordata[wall_force_adr], dtype=np.float64)
        contact_force_mag = float(np.sum(per_wall_forces))

        # curriculum stage gating: 0=reach, 1=grasp, 2=lift, 3=above-hole + aligned
        fingers_on_peg = num_contacts >= 2
        peg_lifted = peg_pos[2] > self._initial_peg_height + 0.02
        peg_near_hole = float(np.linalg.norm(peg_pos[:2] - hole_pos[:2])) < 0.03
        peg_aligned = abs(float(np.dot(peg_axis, hole_axis))) > 0.95

        if num_contacts == 0:
            self._no_contact_grace += 1
        else:
            self._no_contact_grace = 0

        if self._no_contact_grace >= 5:
            self._stage = 0
        else:
            target = 0
            if fingers_on_peg:
                target = 1
            if peg_lifted:
                target = 2
            if peg_near_hole and peg_aligned:
                target = 3
            self._stage = max(self._stage, target)

        # reward
        peg_height = float(peg_pos[2])

        reward, reward_info = self.reward_calculator.compute(
            stage=self._stage,
            finger_positions=finger_pos,
            peg_position=peg_pos,
            peg_axis=peg_axis,
            hole_position=hole_pos,
            hole_axis=hole_axis,
            insertion_depth=insertion_depth,
            contact_force_magnitude=contact_force_mag,
            num_fingers_in_contact=num_contacts,
            contact_finger_indices=contact_finger_indices,
            peg_height=peg_height,
            peg_linvel=peg_linvel,
            actions=action.astype(np.float64),
            previous_actions=self._previous_actions,
        )

        self._previous_actions = action.astype(np.float64).copy()

        peg_length = peg_half_length * 2.0 + peg_radius * 2.0

        insertion_success = bool(
            insertion_depth > self.reward_config.success_threshold * peg_length
            and self.reward_calculator._insertion_hold_steps >= self.reward_config.peg_hold_steps
        )
        fell_off = bool(peg_pos[2] < self.scene_config.table_height - 0.1)

        # success → truncation (bootstrap from terminal obs); fell_off → terminated.
        terminated = fell_off
        truncated = insertion_success and not fell_off

        obs = self._get_obs()
        info = {
            "stage": self._stage,
            **reward_info,
        }
        info["reward/total"] = float(reward)
        info["is_success"] = bool(insertion_success)

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        nm = self.nm

        joint_pos = self.data.qpos[nm.hand_qpos_start : nm.hand_qpos_end]
        joint_vel = self.data.qvel[nm.hand_qvel_start : nm.hand_qvel_end]

        peg_pos, peg_quat, peg_linvel, peg_angvel = get_object_state(
            self.data, nm.peg_body_id, nm.peg_qpos_start, nm.peg_qvel_start
        )

        hole_pos = self.data.xpos[nm.hole_body_id].copy()
        hole_quat = np.zeros(4)
        mujoco.mju_mat2Quat(hole_quat, self.data.xmat[nm.hole_body_id].flatten())

        rel_pos, ang_error = get_peg_hole_relative(self.data, nm.peg_body_id, nm.hole_body_id)

        fingertip_pos = get_fingertip_positions(self.data, nm.fingertip_site_ids)
        fingertip_peg_dist = np.linalg.norm(fingertip_pos - peg_pos, axis=1)

        palm_pos = get_palm_position(self.data, nm.palm_body_id)
        rel_peg_to_palm = peg_pos - palm_pos

        insertion_depth = get_insertion_depth(
            self.data,
            nm.peg_body_id,
            nm.hole_body_id,
            self.scene_config.peg_half_length,
            self.scene_config.peg_radius,
        )

        wall_force_adr = nm.sensor_map.wall_force_adr
        per_wall_forces = np.asarray(self.data.sensordata[wall_force_adr], dtype=np.float64)
        contact_force_mag = float(np.sum(per_wall_forces))

        contact_forces = np.append(per_wall_forces, contact_force_mag)

        obs = np.concatenate(
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
                fingertip_pos.flatten(),
                fingertip_peg_dist,
                rel_peg_to_palm,
                [insertion_depth],
                contact_forces,
                [float(self._stage)],
                self._previous_actions,
            ]
        )

        return obs

    def set_curriculum_params(self, clearance: float, p_pre_grasped: float) -> None:
        self._clearance = clearance
        self._p_pre_grasped = float(p_pre_grasped)

    def render(self) -> np.ndarray | None:  # type: ignore[override]
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)

            self._renderer.update_scene(self.data, camera="track_cam")
            return np.asarray(self._renderer.render())

        elif self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self._viewer.sync()
            return None

        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
