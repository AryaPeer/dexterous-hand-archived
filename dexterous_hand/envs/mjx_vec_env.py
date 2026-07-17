from __future__ import annotations

from typing import Any, NamedTuple

from gymnasium import spaces
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from dexterous_hand.config import DomainRandomization


class DRParams(NamedTuple):
    mass_mult: jnp.ndarray
    friction_mult: jnp.ndarray
    gain_mult: jnp.ndarray


def _apply_dr(mjx_model: Any, dr_params: DRParams) -> Any:
    return mjx_model.replace(
        body_mass=mjx_model.body_mass * dr_params.mass_mult,
        geom_friction=mjx_model.geom_friction.at[:, 0].multiply(dr_params.friction_mult),
        actuator_gainprm=mjx_model.actuator_gainprm.at[:, 0].multiply(dr_params.gain_mult),
    )


class MjxVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        seed: int = 42,
        obs_noise_std: float = 0.0,
        dr: DomainRandomization | None = None,
    ) -> None:
        """Build the batched mjx state, jit/vmap the per-env reset+step+obs functions."""

        self._num_envs = num_envs
        self._seed = seed
        self._obs_noise_std = float(obs_noise_std)
        self._dr_config = dr if dr is not None else DomainRandomization(enabled=False)

        self._mj_model = self._build_model()
        self._mj_data = mujoco.MjData(self._mj_model)
        self._mjx_model = mjx.put_model(self._mj_model)

        n_obs = self._obs_size()
        n_act = self._action_size()

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

        super().__init__(num_envs, observation_space, action_space)

        self._master_key = jax.random.PRNGKey(seed)
        noise_key, self._master_key = jax.random.split(self._master_key)
        self._obs_noise_key = noise_key
        dr_key, self._master_key = jax.random.split(self._master_key)
        self._dr_key = dr_key
        self._env_keys = jax.random.split(self._master_key, num_envs)

        self._ctrl_low = jnp.array(self._mj_model.actuator_ctrlrange[:n_act, 0])
        self._ctrl_high = jnp.array(self._mj_model.actuator_ctrlrange[:n_act, 1])

        self._batched_reset = jax.jit(jax.vmap(self._reset_single, in_axes=(None, 0, 0)))
        self._batched_get_obs = jax.jit(jax.vmap(self._get_obs_single, in_axes=(None, 0, 0)))
        self._batched_step = self._build_batched_step()

        self._mjx_data_batch = None
        self._env_state_batch = None
        self._step_count = jnp.zeros(num_envs, dtype=jnp.int32)
        self._dr_params_batch: DRParams | None = None

        self._pending_obs: np.ndarray | None = None
        self._pending_rewards: np.ndarray | None = None
        self._pending_dones: np.ndarray | None = None
        self._pending_infos: list[dict] | None = None

    def _build_batched_step(self) -> Any:
        def _dr_step(
            mjx_model: Any, dr_params: DRParams, data: Any, state: Any, action: jax.Array
        ) -> Any:
            local_model = _apply_dr(mjx_model, dr_params)
            return self._step_single(local_model, data, state, action)

        return jax.jit(jax.vmap(_dr_step, in_axes=(None, 0, 0, 0, 0)))

    def _build_model(self) -> mujoco.MjModel:
        raise NotImplementedError

    def _reset_single(self, mjx_model: Any, mjx_data: Any, key: jax.Array) -> Any:
        raise NotImplementedError

    def _step_single(self, mjx_model: Any, mjx_data: Any, env_state: Any, action: jax.Array) -> Any:
        raise NotImplementedError

    def _get_obs_single(self, mjx_model: Any, mjx_data: Any, env_state: Any) -> jax.Array:
        raise NotImplementedError

    def _obs_size(self) -> int:
        raise NotImplementedError

    def _action_size(self) -> int:
        raise NotImplementedError

    @property
    def _max_episode_steps(self) -> int:
        raise NotImplementedError

    def _sample_dr_params_single(self, key: jax.Array) -> DRParams:
        nbody = int(self._mj_model.nbody)
        ngeom = int(self._mj_model.ngeom)
        nu = int(self._mj_model.nu)
        if not self._dr_config.enabled:
            return DRParams(
                mass_mult=jnp.ones(nbody),
                friction_mult=jnp.ones(ngeom),
                gain_mult=jnp.ones(nu),
            )
        k_m, k_f, k_g = jax.random.split(key, 3)
        m_lo, m_hi = self._dr_config.mass_range
        f_lo, f_hi = self._dr_config.friction_range
        g_lo, g_hi = self._dr_config.actuator_gain_range
        return DRParams(
            mass_mult=jax.random.uniform(k_m, shape=(nbody,), minval=m_lo, maxval=m_hi),
            friction_mult=jax.random.uniform(k_f, shape=(ngeom,), minval=f_lo, maxval=f_hi),
            gain_mult=jax.random.uniform(k_g, shape=(nu,), minval=g_lo, maxval=g_hi),
        )

    def _sample_dr_params_batch(self, key: jax.Array) -> DRParams:
        keys = jax.random.split(key, self._num_envs)
        return jax.vmap(self._sample_dr_params_single)(keys)

    def _noisy_obs(self, obs: jax.Array) -> np.ndarray:
        # np.array (not np.asarray) — asarray returns a read-only view of the JAX device buffer.
        if self._obs_noise_std <= 0.0:
            return np.array(obs)
        self._obs_noise_key, subkey = jax.random.split(self._obs_noise_key)
        noise = jax.random.normal(subkey, obs.shape) * self._obs_noise_std
        return np.array(obs + noise)

    def reset(self) -> VecEnvObs:
        base_data = mjx.make_data(self._mjx_model)
        batch_data = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self._num_envs,) + x.shape),
            base_data,
        )

        self._env_keys = jax.random.split(jax.random.fold_in(self._master_key, 0), self._num_envs)

        self._dr_key, dr_subkey = jax.random.split(self._dr_key)
        self._dr_params_batch = self._sample_dr_params_batch(dr_subkey)

        batch_data, env_state = self._batched_reset(self._mjx_model, batch_data, self._env_keys)
        self._mjx_data_batch = batch_data
        self._env_state_batch = env_state
        self._step_count = jnp.zeros(self._num_envs, dtype=jnp.int32)

        obs = self._batched_get_obs(self._mjx_model, batch_data, env_state)
        return self._noisy_obs(obs)

    def step_async(self, actions: np.ndarray) -> None:
        actions_jax = jnp.array(actions, dtype=jnp.float32)

        new_data, new_state, obs, rewards, dones, reward_info = self._batched_step(
            self._mjx_model,
            self._dr_params_batch,
            self._mjx_data_batch,
            self._env_state_batch,
            actions_jax,
        )

        self._step_count = self._step_count + 1
        timed_out = self._step_count >= self._max_episode_steps

        # NaN guard: blown-up envs would poison the policy via reward grads — reset and zero them.
        bad = jnp.any(jnp.isnan(new_data.qpos), axis=1) | jnp.any(jnp.isnan(obs), axis=1)
        rewards = jnp.where(bad, 0.0, jnp.where(jnp.isnan(rewards), 0.0, rewards))
        obs = jnp.where(bad[:, None], 0.0, jnp.nan_to_num(obs, nan=0.0))
        if reward_info is not None:
            reward_info["metrics/nan_rate"] = bad.astype(jnp.float32)

        # Bootstrap (TimeLimit.truncated) ONLY on timeout: the episode was cut
        # short by the horizon but would have continued, so SB3 adds
        # gamma*V(terminal_obs). Physical failures (fell/launched) are true
        # (absorbing) terminals — no bootstrap. Since 2026-07-14 neither task
        # terminates on success: the solved state is the highest-paying
        # per-step state and the episode runs the horizon (holding the cube at
        # height / peg settled in the bore), which removes the
        # success-termination-farming incentive. is_success still propagates
        # to per-env infos via reward_info (logging + SB3 success-rate
        # tracking).
        truncated_only = timed_out & ~dones
        dones = dones | timed_out | bad

        needs_reset = dones
        if jnp.any(needs_reset):
            self._env_keys = jax.vmap(
                lambda k, need: jax.lax.cond(
                    need, lambda k: jax.random.split(k)[0], lambda k: k, k
                ),
                in_axes=(0, 0),
            )(self._env_keys, needs_reset)

            reset_data, reset_state = self._batched_reset(self._mjx_model, new_data, self._env_keys)

            # resample DR multipliers only for the envs being reset — others keep theirs.
            self._dr_key, dr_subkey = jax.random.split(self._dr_key)
            reset_dr = self._sample_dr_params_batch(dr_subkey)
            self._dr_params_batch = jax.tree.map(
                lambda r, n: jnp.where(
                    needs_reset.reshape(-1, *([1] * (r.ndim - 1))) if r.ndim > 1 else needs_reset,
                    r,
                    n,
                ),
                reset_dr,
                self._dr_params_batch,
            )

            new_data = jax.tree.map(
                lambda r, n: jnp.where(needs_reset.reshape(-1, *([1] * (r.ndim - 1))), r, n),
                reset_data,
                new_data,
            )
            new_state = jax.tree.map(
                lambda r, n: jnp.where(
                    needs_reset.reshape(-1, *([1] * (r.ndim - 1))) if r.ndim > 0 else needs_reset,
                    r,
                    n,
                ),
                reset_state,
                new_state,
            )

            self._step_count = jnp.where(needs_reset, 0, self._step_count)

            reset_obs = self._batched_get_obs(self._mjx_model, new_data, new_state)
            obs_np = self._noisy_obs(obs)
            reset_obs_np = self._noisy_obs(reset_obs)
        else:
            obs_np = self._noisy_obs(obs)
            reset_obs_np = None

        self._mjx_data_batch = new_data
        self._env_state_batch = new_state

        dones_np = np.asarray(dones)
        truncated_np = np.asarray(truncated_only)
        rewards_np = np.asarray(rewards, dtype=np.float64)

        # Host-side info cost dominates the python loop at large num_envs
        # (num_envs dicts x ~25 numpy scalars per control step), so the
        # reward/metric streams are reduced to batch MEANS on device and
        # attached to infos[0] only — the logging/gate callbacks read them
        # from whichever info carries them, and a mean-of-per-step-means
        # equals the mean over envs x steps. Only `is_success` stays per-env
        # (SB3's success-rate tracking is per-episode).
        is_success_np: np.ndarray | None = None
        agg_info: dict[str, float] = {}
        if reward_info is not None:
            per_env = reward_info.pop("is_success", None)
            if per_env is not None:
                is_success_np = np.asarray(per_env)
            keys = sorted(reward_info.keys())
            if keys:
                means = np.asarray(jnp.stack([jnp.mean(reward_info[k]) for k in keys]))
                means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
                agg_info = {k: float(m) for k, m in zip(keys, means, strict=True)}

        infos: list[dict[str, Any]] = []
        for i in range(self._num_envs):
            info: dict[str, Any] = {}
            if is_success_np is not None:
                info["is_success"] = float(is_success_np[i])
            if dones_np[i]:
                info["terminal_observation"] = obs_np[i].copy()
                if truncated_np[i]:
                    info["TimeLimit.truncated"] = True
                if reset_obs_np is not None:
                    obs_np[i] = reset_obs_np[i]
            infos.append(info)
        infos[0].update(agg_info)

        self._pending_obs = obs_np
        self._pending_rewards = rewards_np
        self._pending_dones = dones_np
        self._pending_infos = infos

    def step_wait(self) -> VecEnvStepReturn:
        assert self._pending_obs is not None
        obs = self._pending_obs
        rewards = self._pending_rewards
        dones = self._pending_dones
        infos = self._pending_infos

        self._pending_obs = None
        self._pending_rewards = None
        self._pending_dones = None
        self._pending_infos = None

        return obs, rewards, dones, infos  # type: ignore[return-value]

    def close(self) -> None:
        pass

    def env_is_wrapped(self, wrapper_class: Any, indices: Any = None) -> list[bool]:  # type: ignore[override]
        del wrapper_class, indices
        return [False] * self._num_envs

    def env_method(  # type: ignore[override]
        self,
        method_name: str,
        *method_args: Any,
        indices: Any = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        if indices is None:
            indices = range(self._num_envs)

        if hasattr(self, method_name):
            fn = getattr(self, method_name)
            return [fn(*method_args, **method_kwargs)] * len(list(indices))

        return [None] * len(list(indices))

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:  # type: ignore[override]
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            n = self._num_envs if indices is None else len(indices)
            return [val] * n
        return [None] * (self._num_envs if indices is None else len(indices))

    def set_attr(  # type: ignore[override]
        self, attr_name: str, value: Any, indices: Any = None
    ) -> None:
        setattr(self, attr_name, value)

    def seed(self, seed: int | None = None) -> None:  # type: ignore[override]
        if seed is not None:
            self._master_key = jax.random.PRNGKey(seed)
