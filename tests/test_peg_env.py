
import numpy as np
import pytest

pytest.importorskip("mujoco.mjx")
pytest.importorskip("jax")

from dexterous_hand.config import MjxPegTrainConfig  # noqa: E402
from dexterous_hand.envs.peg_env import ShadowHandPegMjxEnv  # noqa: E402


@pytest.mark.slow
class TestPegMjxSmoke:
    def test_reset_and_step(self):
        env = ShadowHandPegMjxEnv(num_envs=4, seed=0, max_episode_steps=50)
        try:
            obs = env.reset()
            assert obs.shape == (4, 134)
            assert np.all(np.isfinite(obs))

            actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
            for _ in range(5):
                env.step_async(actions)
                obs, rewards, dones, infos = env.step_wait()
                assert obs.shape == (4, 134)
                assert rewards.shape == (4,)
                assert dones.shape == (4,)
                assert len(infos) == 4
                assert np.all(np.isfinite(obs))
                assert np.all(np.isfinite(rewards))
        finally:
            env.close()

    def test_pregrasped_lift_reference_clamped_to_table(self):
        """Round-17 regression: pre-grasped spawns place the peg in-hand
        (~0.52), and referencing lift from THERE paid `lift` 0 at the engaged
        pose — the per-step chain inverted (held-high 25.5 > hover 23.5 >
        engaged 18.7) and pushed the policy away from the endgame in 100% of
        early-curriculum episodes. The reference must clamp to the table
        spawn height."""
        env = ShadowHandPegMjxEnv(num_envs=4, seed=0, max_episode_steps=50)
        try:
            env.set_curriculum_params(
                clearance=float(env.scene_config.clearance), p_pre_grasped=1.0
            )
            env.reset()
            cfg = env.scene_config
            table_spawn = (
                cfg.table_height + cfg.peg_half_length + cfg.peg_radius + 0.001
            )
            init_h = np.asarray(env._env_state_batch.initial_peg_height)
            assert np.all(init_h <= table_spawn + 1e-6)
            reward_init_h = np.asarray(
                env._env_state_batch.reward_state.initial_peg_height
            )
            assert np.all(reward_init_h <= table_spawn + 1e-6)
            # the clamp must have had something to clamp: pre-grasped pegs sit
            # in-hand well above the table spawn height after settle
            peg_z = np.asarray(env._mjx_data_batch.xpos[:, env._nm.peg_body_id, 2])
            assert np.all(peg_z > table_spawn + 0.02)
        finally:
            env.close()

    def test_from_config_seeds_first_rollout_p_pre_grasped(self):
        """SB3 resets envs before the curriculum callback's
        _on_training_start fires, so the env must be BORN with curriculum
        stage 0's p_pre_grasped — not 0.0 — or the whole first episode wave
        is table-spawned regardless of the schedule."""
        config = MjxPegTrainConfig(num_envs=2, max_episode_steps=10)
        env = ShadowHandPegMjxEnv.from_config(config)
        try:
            expected = config.curriculum_stages[0][2]
            assert float(env._p_pre_grasped) == pytest.approx(expected)
        finally:
            env.close()
