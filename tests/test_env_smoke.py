import gymnasium as gym
import numpy as np
import pytest

import dexterous_hand.envs  # noqa: F401 — triggers env registration

ENV_IDS = [
    ("ShadowHandGrasp-v0", 105),
    ("ShadowHandReorient-v0", 109),
    ("ShadowHandPeg-v0", 134),
]


@pytest.mark.slow
@pytest.mark.parametrize(("env_id", "expected_obs_size"), ENV_IDS)
def test_env_smoke(env_id: str, expected_obs_size: int) -> None:
    # reset + 100 random-action steps; asserts obs shape, finite reward,
    # bool done, auto-reset on termination. this is the minimal pre-flight
    # any code change on the reward/obs/scene side should clear.
    env = gym.make(env_id)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (expected_obs_size,), (
            f"{env_id}: obs shape {obs.shape} != declared ({expected_obs_size},)"
        )
        assert env.observation_space.shape == (expected_obs_size,)
        assert np.isfinite(obs).all()

        rng = np.random.default_rng(0)
        for _ in range(100):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (expected_obs_size,)
            assert np.isfinite(obs).all(), f"{env_id}: NaN/inf in obs"
            assert np.isfinite(reward), f"{env_id}: NaN/inf reward"
            assert bool(terminated) == terminated  # accepts bool or np.bool_
            assert bool(truncated) == truncated
            if terminated or truncated:
                obs, info = env.reset()
                assert obs.shape == (expected_obs_size,)
    finally:
        env.close()
