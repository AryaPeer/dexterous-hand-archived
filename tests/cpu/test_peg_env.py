import numpy as np
import pytest

import dexterous_hand.envs  # noqa: F401


@pytest.mark.slow
class TestPegEnvSpaces:
    def test_observation_shape(self, peg_env):
        assert peg_env.observation_space.shape == (134,)

    def test_action_shape(self, peg_env):
        assert peg_env.action_space.shape == (23,)

    def test_action_bounds(self, peg_env):
        assert float(peg_env.action_space.low.min()) == -1.0
        assert float(peg_env.action_space.high.max()) == 1.0

@pytest.mark.slow
class TestPegEnvReset:
    def test_reset_obs_shape(self, peg_env):
        obs, _ = peg_env.reset(seed=42)
        assert obs.shape == (134,)

    def test_reset_obs_finite(self, peg_env):
        obs, _ = peg_env.reset(seed=42)
        assert np.all(np.isfinite(obs))

    def test_deterministic_seeding(self, peg_env):
        obs1, _ = peg_env.reset(seed=123)
        obs2, _ = peg_env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

@pytest.mark.slow
class TestPegEnvStep:
    def test_step_returns(self, peg_env):
        peg_env.reset(seed=42)
        action = peg_env.action_space.sample()
        obs, reward, terminated, truncated, info = peg_env.step(action)
        assert obs.shape == (134,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_obs_finite_after_many_steps(self, peg_env):
        peg_env.reset(seed=42)
        for i in range(200):
            action = peg_env.action_space.sample()
            obs, _, terminated, truncated, _ = peg_env.step(action)
            assert np.all(np.isfinite(obs)), f"Non-finite at step {i}"
            if terminated or truncated:
                peg_env.reset()

    def test_reward_info_keys(self, peg_env):
        peg_env.reset(seed=42)
        _, _, _, _, info = peg_env.step(peg_env.action_space.sample())
        expected = [
            "reward/reach",
            "reward/grasp",
            "reward/lift",
            "reward/align",
            "reward/depth",
            "reward/complete",
            "reward/force_penalty",
            "reward/drop",
            "reward/action_penalty",
            "reward/total",
        ]
        for key in expected:
            assert key in info, f"Missing key: {key}"

    def test_metric_keys(self, peg_env):
        peg_env.reset(seed=42)
        _, _, _, _, info = peg_env.step(peg_env.action_space.sample())
        for key in [
            "metrics/stage",
            "metrics/num_finger_contacts",
            "metrics/peg_height",
            "metrics/insertion_depth",
            "metrics/contact_force",
            "metrics/lateral_distance",
            "metrics/insertion_hold_steps",
        ]:
            assert key in info, f"Missing key: {key}"

    def test_reward_reasonable(self, peg_env):
        peg_env.reset(seed=42)
        for _ in range(50):
            _, reward, term, trunc, _ = peg_env.step(peg_env.action_space.sample())
            assert np.isfinite(reward)
            assert abs(reward) < 1000
            if term or trunc:
                peg_env.reset()

@pytest.mark.slow
class TestPegStages:
    def test_stage_in_info(self, peg_env):
        peg_env.reset(seed=42)
        _, _, _, _, info = peg_env.step(peg_env.action_space.sample())
        assert "metrics/stage" in info
        assert 0 <= info["metrics/stage"] <= 3

@pytest.mark.slow
class TestPegCurriculum:
    def test_set_curriculum_params(self, fresh_peg_env):
        fresh_peg_env.unwrapped.set_curriculum_params(0.002, 0.0)
        assert fresh_peg_env.unwrapped._clearance == 0.002
        assert fresh_peg_env.unwrapped._p_pre_grasped == 0.0
        obs, _ = fresh_peg_env.reset()
        assert np.all(np.isfinite(obs))

    def test_pre_grasped_mode(self, fresh_peg_env):
        fresh_peg_env.unwrapped.set_curriculum_params(0.004, 1.0)
        assert fresh_peg_env.unwrapped._p_pre_grasped == 1.0
        obs, _ = fresh_peg_env.reset()
        assert np.all(np.isfinite(obs))

@pytest.mark.slow
class TestPegEnvRender:
    def test_rgb_render(self, peg_env):
        peg_env.reset(seed=0)
        frame = peg_env.render()
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3       

@pytest.mark.slow
class TestPegMultipleResets:
    def test_100_resets(self, peg_env):
        for i in range(100):
            obs, _ = peg_env.reset(seed=i)
            assert np.all(np.isfinite(obs))
