
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from dexterous_hand.config import PegRewardConfig, RewardConfig  # noqa: E402
from dexterous_hand.rewards.grasp_reward import (  # noqa: E402
    grasp_reward,
    init_grasp_reward_state,
)
from dexterous_hand.rewards.peg_reward import (  # noqa: E402
    init_peg_reward_state,
    peg_reward,
)


def _grasp_kwargs(cfg: RewardConfig, table_height: float) -> dict:
    return dict(
        finger_positions=jnp.zeros((5, 3)),
        object_position=jnp.array([0.0, 0.0, 0.5]),
        object_linear_velocity=jnp.zeros(3),
        finger_contact_mask=jnp.array([True, True, True, False, False]),
        actions=jnp.zeros(23),
        previous_actions=jnp.zeros(23),
        table_height=table_height,
        lift_target=cfg.lift_target,
        hold_velocity_threshold=cfg.hold_velocity_threshold,
        drop_penalty_value=cfg.drop_penalty,
        no_contact_idle_penalty=cfg.no_contact_idle_penalty,
        success_bonus_per_step=cfg.success_bonus_per_step,
        success_hold_steps=cfg.success_hold_steps,
        weights=cfg.weights,
        reach_tanh_k=cfg.reach_tanh_k,
        hold_height_k=cfg.hold_height_smoothness_k,
        hold_velocity_k=cfg.hold_velocity_smoothness_k,
        fingertip_weights=cfg.fingertip_weights,
        drop_arm_height=cfg.drop_arm_height,
        action_penalty_scale=cfg.action_penalty_scale,
    )

class TestGraspJax:
    def test_jit_compiles(self):
        cfg = RewardConfig()

                                                                            
                                                          
        @jax.jit
        def _run(
            state,
            finger_positions,
            object_position,
            object_linear_velocity,
            finger_contact_mask,
            actions,
            previous_actions,
        ):
            return grasp_reward(
                state=state,
                finger_positions=finger_positions,
                object_position=object_position,
                object_linear_velocity=object_linear_velocity,
                finger_contact_mask=finger_contact_mask,
                actions=actions,
                previous_actions=previous_actions,
                table_height=0.4,
                lift_target=cfg.lift_target,
                hold_velocity_threshold=cfg.hold_velocity_threshold,
                drop_penalty_value=cfg.drop_penalty,
                no_contact_idle_penalty=cfg.no_contact_idle_penalty,
                success_bonus_per_step=cfg.success_bonus_per_step,
                success_hold_steps=cfg.success_hold_steps,
                weights=cfg.weights,
                reach_tanh_k=cfg.reach_tanh_k,
                hold_height_k=cfg.hold_height_smoothness_k,
                hold_velocity_k=cfg.hold_velocity_smoothness_k,
                fingertip_weights=cfg.fingertip_weights,
                drop_arm_height=cfg.drop_arm_height,
                action_penalty_scale=cfg.action_penalty_scale,
            )

        state = init_grasp_reward_state(0.4, 0.4)
        total, _, info = _run(
            state,
            jnp.zeros((5, 3)),
            jnp.array([0.0, 0.0, 0.5]),
            jnp.zeros(3),
            jnp.array([True, True, True, False, False]),
            jnp.zeros(23),
            jnp.zeros(23),
        )
        assert np.isfinite(float(total))
        assert "reward/total" in info

    def test_vmap_over_batch(self):
        cfg = RewardConfig()
        B = 8

        def _one(state, obj_z):
            kw = _grasp_kwargs(cfg, 0.4)
            kw["object_position"] = jnp.array([0.0, 0.0, obj_z])
            total, _, _ = grasp_reward(state=state, **kw)
            return total

        batched = jax.vmap(_one, in_axes=(0, 0))
        states = jax.vmap(lambda z: init_grasp_reward_state(z, 0.4))(jnp.linspace(0.4, 0.6, B))
        totals = batched(states, jnp.linspace(0.4, 0.6, B))
        assert totals.shape == (B,)
        assert bool(jnp.all(jnp.isfinite(totals)))

    def test_lifted_latch_advances(self):
        cfg = RewardConfig()
        kw = _grasp_kwargs(cfg, 0.4)
        kw["object_position"] = jnp.array([0.0, 0.0, 0.55])                     
        state = init_grasp_reward_state(0.4, 0.4)
        _, new_state, _ = grasp_reward(state=state, **kw)
        assert bool(new_state.was_lifted)

        kw["object_position"] = jnp.array([0.0, 0.0, 0.405])
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        _, final_state, info = grasp_reward(state=new_state, **kw)

        assert float(info["reward/drop"]) < 0.0

    def test_success_annuity_pays_while_held(self):
        """Success pays per-step while the hold condition is sustained and
        stops the moment the cube is dropped. A lift/hold/drop cycle therefore
        strictly loses income vs steady holding (it forfeits the payment
        stream for success_hold_steps + the -20 drop penalty), so there is no
        yo-yo farming exploit for a latch to patch."""
        cfg = RewardConfig()
        held = _grasp_kwargs(cfg, 0.4)
        # 3 contacts (default mask), at rest, lift_factor 1.0 -> at_target
        held["object_position"] = jnp.array([0.0, 0.0, 0.55])
        dropped = _grasp_kwargs(cfg, 0.4)
        dropped["object_position"] = jnp.array([0.0, 0.0, 0.436])
        dropped["finger_contact_mask"] = jnp.array([False] * 5)

        state = init_grasp_reward_state(0.436, 0.4)
        bonuses = []
        for _ in range(cfg.success_hold_steps + 5):
            _, state, info = grasp_reward(state=state, **held)
            bonuses.append(float(info["reward/success"]))
        # nothing until the hold requirement is met, then the annuity every step
        assert all(b == 0.0 for b in bonuses[: cfg.success_hold_steps - 1])
        assert all(b == cfg.success_bonus_per_step for b in bonuses[cfg.success_hold_steps - 1 :])

        # dropping stops the payment immediately and charges the drop penalty
        _, state, info = grasp_reward(state=state, **dropped)
        assert float(info["reward/success"]) == 0.0
        assert float(info["reward/drop"]) < 0.0
        for _ in range(2):
            _, state, info = grasp_reward(state=state, **dropped)
            assert float(info["reward/success"]) == 0.0

        # a re-hold has to re-earn the full hold window before payment resumes
        rehold_bonuses = []
        for _ in range(cfg.success_hold_steps + 5):
            _, state, info = grasp_reward(state=state, **held)
            rehold_bonuses.append(float(info["reward/success"]))
        assert all(b == 0.0 for b in rehold_bonuses[: cfg.success_hold_steps - 1])
        assert all(
            b == cfg.success_bonus_per_step
            for b in rehold_bonuses[cfg.success_hold_steps - 1 :]
        )
        assert float(info["is_success"]) == 1.0


class TestPegJax:
    def _kw(self) -> dict:
        cfg = PegRewardConfig()
        peg_length = 0.06
        return dict(
            stage=jnp.asarray(0),
            finger_positions=jnp.zeros((5, 3)),
            peg_position=jnp.array([0.0, 0.0, 0.9]),
            peg_axis=jnp.array([0.0, 0.0, 1.0]),
            hole_position=jnp.array([0.0, 0.0, 0.88]),
            hole_axis=jnp.array([0.0, 0.0, 1.0]),
            insertion_depth=jnp.asarray(0.0),
            contact_force_magnitude=jnp.asarray(0.0),
            finger_contact_mask=jnp.array([True, True, True, False, False]),
            peg_height=jnp.asarray(0.9),
            peg_linvel=jnp.zeros(3),
            actions=jnp.zeros(22),
            previous_actions=jnp.zeros(22),
            weights=cfg.weights,
            peg_length=peg_length,
            lift_target=cfg.lift_target,
            table_height=0.82,
            drop_penalty_value=cfg.drop_penalty,
            complete_bonus=cfg.complete_bonus,
            force_threshold=cfg.force_threshold,
            idle_stage0_penalty=cfg.idle_stage0_penalty,
            lateral_gate_k=cfg.lateral_gate_k,
            idle_stage_cutoff=cfg.idle_stage_cutoff,
            success_threshold=cfg.success_threshold,
            peg_hold_steps=cfg.peg_hold_steps,
            reach_tanh_k=cfg.reach_tanh_k,
            fingertip_weights=cfg.fingertip_weights,
        )

    def test_jit_compiles(self):
        cfg = PegRewardConfig()
        peg_length = 0.06

        @jax.jit
        def _run(
            state,
            stage,
            finger_positions,
            peg_position,
            peg_axis,
            hole_position,
            hole_axis,
            insertion_depth,
            contact_force_magnitude,
            finger_contact_mask,
            peg_height,
            peg_linvel,
            actions,
            previous_actions,
        ):
            return peg_reward(
                state=state,
                stage=stage,
                finger_positions=finger_positions,
                peg_position=peg_position,
                peg_axis=peg_axis,
                hole_position=hole_position,
                hole_axis=hole_axis,
                insertion_depth=insertion_depth,
                contact_force_magnitude=contact_force_magnitude,
                finger_contact_mask=finger_contact_mask,
                peg_height=peg_height,
                peg_linvel=peg_linvel,
                actions=actions,
                previous_actions=previous_actions,
                weights=cfg.weights,
                peg_length=peg_length,
                lift_target=cfg.lift_target,
                table_height=0.82,
                drop_penalty_value=cfg.drop_penalty,
                complete_bonus=cfg.complete_bonus,
                force_threshold=cfg.force_threshold,
                idle_stage0_penalty=cfg.idle_stage0_penalty,
                lateral_gate_k=cfg.lateral_gate_k,
                idle_stage_cutoff=cfg.idle_stage_cutoff,
                success_threshold=cfg.success_threshold,
                peg_hold_steps=cfg.peg_hold_steps,
                reach_tanh_k=cfg.reach_tanh_k,
                fingertip_weights=cfg.fingertip_weights,
            )

        state = init_peg_reward_state(0.85)
        total, _, info = _run(
            state,
            jnp.asarray(0),
            jnp.zeros((5, 3)),
            jnp.array([0.0, 0.0, 0.9]),
            jnp.array([0.0, 0.0, 1.0]),
            jnp.array([0.0, 0.0, 0.88]),
            jnp.array([0.0, 0.0, 1.0]),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.array([True, True, True, False, False]),
            jnp.asarray(0.9),
            jnp.zeros(3),
            jnp.zeros(22),
            jnp.zeros(22),
        )
        assert np.isfinite(float(total))
        assert "reward/depth" in info

    def test_insertion_hold_smoothly_grows_complete_bonus(self):
        # smooth bonus replaces the binary cliff: complete = bonus *
        # sigmoid(20*(frac-0.7)) * sigmoid(hold/5 - 1). Below threshold
        # complete is near-zero; once frac > threshold complete grows
        # monotonically with hold count and asymptotes near
        # complete_bonus * sigmoid(20*(0.917-0.7)) ≈ 0.987 * complete_bonus.
        cfg = PegRewardConfig()
        kw = self._kw()
        kw["insertion_depth"] = jnp.asarray(0.055)
        state = init_peg_reward_state(0.85)

        prev_complete = -1.0
        for _ in range(50):
            _, state, info = peg_reward(state=state, **kw)
            value = float(info["reward/complete"])
            assert value >= prev_complete - 1e-6
            prev_complete = value

        # insertion_fraction = 0.055 / 0.06 = 0.9166...; excess over 0.7
        # is ≈ 0.2166. asymptote = bonus * sigmoid(20 * 0.2166).
        excess = 0.055 / 0.06 - cfg.success_threshold
        asymptote = cfg.complete_bonus / (1.0 + float(np.exp(-20.0 * excess)))
        np.testing.assert_allclose(prev_complete, asymptote, rtol=1e-3)

    def test_complete_pays_with_zero_contacts(self):
        # Regression for the 2026-07-14 release-forfeit bug: the bore cannot
        # admit fingers, so success REQUIRES releasing the peg — complete was
        # gated on contact_scale and paid ~0 for a settled released peg while
        # a gripped hover below threshold farmed shaping forever.
        kw = self._kw()
        kw["insertion_depth"] = jnp.asarray(0.055)
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        state = init_peg_reward_state(0.85)
        value = 0.0
        for _ in range(30):
            _, state, info = peg_reward(state=state, **kw)
            value = float(info["reward/complete"])
        assert value > 100.0, (
            f"released settled peg earns complete={value:.1f} — the completion "
            f"payment must not require finger contact"
        )

    def test_place_monotone_toward_engaged_pose(self):
        # place must pay more the closer the peg pose is to the engaged
        # release pose (vertical, tip |release_height| inside the bore).
        cfg = PegRewardConfig()
        peg_length = 0.06
        hole_z = 0.88

        def place_at(peg_xy, peg_z):
            kw = self._kw()
            kw["peg_position"] = jnp.array([peg_xy[0], peg_xy[1], peg_z])
            _, _, info = peg_reward(state=init_peg_reward_state(0.85), **kw)
            return float(info["reward/place"])

        engaged_z = hole_z + cfg.release_height + peg_length / 2.0
        p_table = place_at((0.05, 0.0), 0.85)
        p_lifted = place_at((0.05, 0.0), 0.90)
        p_engaged = place_at((0.0, 0.0), engaged_z)
        assert p_table < p_engaged
        assert p_lifted < p_engaged
        assert p_engaged > 0.9

    def test_drop_penalty_not_fired_when_inserted(self):
        # Releasing the peg INTO the bore is the winning move; the was_lifted
        # drop penalty must never fire on an inserted peg even if its height
        # ends near the spawn height.
        kw = self._kw()
        state = init_peg_reward_state(0.85)
        state = state._replace(was_lifted=jnp.array(True))
        kw["peg_position"] = jnp.array([0.0, 0.0, 0.855])  # lift_height ~0.005
        kw["peg_height"] = jnp.asarray(0.855)
        kw["insertion_depth"] = jnp.asarray(0.045)  # fraction 0.75
        kw["finger_contact_mask"] = jnp.array([False] * 5)
        _, _, info = peg_reward(state=state, **kw)
        assert float(info["reward/drop"]) == 0.0

    def test_complete_below_threshold_stays_small(self):
        # below the success_threshold (0.7), the insertion-fraction sigmoid
        # collapses the bonus toward zero independent of hold count.
        cfg = PegRewardConfig()
        kw = self._kw()
        kw["insertion_depth"] = jnp.asarray(0.03)  # frac=0.5 < 0.7
        state = init_peg_reward_state(0.85)
        for _ in range(20):
            _, state, info = peg_reward(state=state, **kw)
        # sigmoid(20*(0.5-0.7)) = sigmoid(-4) ≈ 0.018; bound check
        assert float(info["reward/complete"]) < 0.05 * cfg.complete_bonus

    def test_depth_reward_clamped(self):
        kw = self._kw()
                                                      
        kw["insertion_depth"] = jnp.asarray(0.5)
        state = init_peg_reward_state(0.85)
        _, _, info = peg_reward(state=state, **kw)
                                                                                  
        assert float(info["reward/depth"]) <= 10.0 + 1e-6

                                          
        kw["insertion_depth"] = jnp.asarray(-0.1)
        _, _, info_neg = peg_reward(state=state, **kw)
        assert float(info_neg["reward/depth"]) == 0.0
