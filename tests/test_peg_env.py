
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

    def test_curriculum_clearance_change_rebuilds_and_steps(self):
        env = ShadowHandPegMjxEnv(num_envs=2, seed=0, max_episode_steps=2)
        try:
            env.reset()
            env.set_curriculum_params(clearance=0.003, p_pre_grasped=0.5)
            assert abs(float(env.scene_config.clearance) - 0.003) < 1e-12
            actions = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
            for step in range(1, 5):
                env.step_async(actions)
                obs, rewards, dones, infos = env.step_wait()
                assert np.all(np.isfinite(obs))
                assert np.all(np.isfinite(rewards))
                assert bool(dones.all()) == (step % 2 == 0)
        finally:
            env.close()

    def test_auto_reset_cycles_episodes(self):
        env = ShadowHandPegMjxEnv(num_envs=4, seed=0, max_episode_steps=3)
        try:
            env.reset()
            actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
            for step in range(1, 9):
                env.step_async(actions)
                obs, _rewards, dones, infos = env.step_wait()
                expect_done = step % 3 == 0
                assert bool(dones.all()) == expect_done
                assert bool(dones.any()) == expect_done
                for i, info in enumerate(infos):
                    if expect_done:
                        assert info["TimeLimit.truncated"] is True
                        assert "is_success" in info
                        term = info["terminal_observation"]
                        assert term.shape == obs[i].shape
                        assert np.all(np.isfinite(term))
                        assert not np.allclose(term, obs[i])
                    else:
                        assert "terminal_observation" not in info
                        assert "is_success" not in info
                assert np.all(np.isfinite(obs))
        finally:
            env.close()

    def test_pregrasped_lift_reference_clamped_to_table(self):
        """Pre-grasped spawns must clamp the lift reference to the table height."""
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
            reward_init_h = np.asarray(
                env._env_state_batch.reward_state.initial_peg_height
            )
            assert np.all(reward_init_h <= table_spawn + 1e-6)
            peg_z = np.asarray(env._mjx_data_batch.xpos[:, env._nm.peg_body_id, 2])
            assert np.all(peg_z > table_spawn + 0.02)
        finally:
            env.close()

    def test_from_config_seeds_first_rollout_p_pre_grasped(self):
        """SB3 resets envs before the curriculum callback's"""
        config = MjxPegTrainConfig(num_envs=2, max_episode_steps=10)
        env = ShadowHandPegMjxEnv.from_config(config)
        try:
            expected = config.curriculum_stages[0][2]
            assert float(env._p_pre_grasped) == pytest.approx(expected)
        finally:
            env.close()
