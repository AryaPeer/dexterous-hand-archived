
import numpy as np
import pytest

pytest.importorskip("mujoco.mjx")
pytest.importorskip("jax")

from dexterous_hand.envs.grasp_env import ShadowHandGraspMjxEnv  # noqa: E402


@pytest.mark.slow
class TestGraspMjxSmoke:
    def test_reset_and_step(self):
        env = ShadowHandGraspMjxEnv(num_envs=4, seed=0, max_episode_steps=50)
        try:
            obs = env.reset()
            assert obs.shape == (4, 108)
            assert np.all(np.isfinite(obs))

            actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
            for _ in range(5):
                env.step_async(actions)
                obs, rewards, dones, infos = env.step_wait()
                assert obs.shape == (4, 108)
                assert rewards.shape == (4,)
                assert dones.shape == (4,)
                assert len(infos) == 4
                assert np.all(np.isfinite(obs))
                assert np.all(np.isfinite(rewards))
        finally:
            env.close()
