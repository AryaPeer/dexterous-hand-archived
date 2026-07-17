from dexterous_hand.config import (
    MjxGraspTrainConfig,
    MjxPegTrainConfig,
    PegRewardConfig,
    PegRewardWeights,
    PegSceneConfig,
    RewardConfig,
    RewardWeights,
    SceneConfig,
)


class TestConfigDefaults:
    def test_scene_config(self):
        c = SceneConfig()
        assert c.mount_x == -0.10
        assert c.mount_y == 0.0
        assert c.mount_height == 0.78
        # 0.005 x 8 = 25 Hz control (Playground-style MJX physics config)
        assert c.sim_timestep == 0.005
        assert c.frame_skip == 8
        assert c.solver_iterations == 8
        assert c.ls_iterations == 8

    def test_reward_weights(self):
        w = RewardWeights()
        assert w.reaching == 0.5
        assert w.grasping == 2.5
        assert w.lifting == 6.0
        assert w.holding == 6.0

    def test_reward_config(self):
        c = RewardConfig()
        assert isinstance(c.weights, RewardWeights)
        # 0.10 = a real, visible pick-up; guards against eroding the lift bar
        # to match a broken scene instead of fixing the scene.
        assert c.lift_target == 0.10
        assert c.hold_height_smoothness_k == 50.0
        assert c.hold_velocity_smoothness_k == 100.0
        assert c.no_contact_idle_penalty == -0.08
        # success pays per-step while held (annuity), never as a one-shot spike
        assert c.success_bonus_per_step == 5.0
        assert c.drop_arm_height == 0.04
        # reach weighting emphasizes the thumb (site order [ff, mf, rf, lf, th])
        assert c.fingertip_weights[4] == max(c.fingertip_weights)

    def test_peg_scene_config(self):
        c = PegSceneConfig()
        assert c.mount_x == -0.10
        assert c.mount_y == 0.0
        assert c.mount_height == 0.82
        assert c.action_smoothing_alpha == 0.2
        assert c.spawn_min_radius == 0.04
        assert c.clearance == 0.004
        assert c.hole_depth == 0.06
        assert c.hole_top_above_table == 0.08
        assert len(c.hole_offset) == 2
        assert c.peg_radius == 0.008
        assert c.peg_half_length == 0.03
        assert c.peg_mass == 0.02
        assert c.solver_iterations == 8
        assert c.ls_iterations == 8

    def test_peg_reward_config(self):
        c = PegRewardConfig()
        assert c.complete_bonus == 250.0
        assert c.force_threshold == 15.0
        assert c.idle_stage0_penalty == -0.3
        assert c.weights.opposition == 1.0
        assert c.weights.axis_in_grip == 1.0
        # place: keypoint shaping to the ENGAGED release pose (2026-07-14) —
        # the only x/y gradient between lift saturation and in-bore depth.
        assert c.weights.place == 8.0
        assert c.release_height == -0.015
        assert c.lateral_gate_k == 5.0
        assert c.peg_hold_steps == 10
        assert c.success_threshold == 0.7

    def test_mjx_peg_curriculum_stages(self):
        c = MjxPegTrainConfig()
        assert isinstance(c.scene_config, PegSceneConfig)
        assert isinstance(c.reward_config, PegRewardConfig)
        assert len(c.curriculum_stages) == 5
        for stage in c.curriculum_stages:
            assert len(stage) == 3
            step, clearance, p = stage
            assert 0.0 <= p <= 1.0

    def test_mjx_log_std_clamp_defaults(self):
        # Bounds must be valid (min < max) and the init must sit inside the
        # range; the clamped Actor only blocks the runaway if the policy
        # actually starts within the allowed range.
        for cls in (MjxGraspTrainConfig, MjxPegTrainConfig):
            c = cls()
            assert c.log_std_min < c.log_std_max
            assert c.log_std_min <= c.log_std_init <= c.log_std_max
            assert c.log_std_max == 0.0

    def test_all_configs_instantiate(self):

        configs = [
            SceneConfig,
            RewardWeights,
            RewardConfig,
            MjxGraspTrainConfig,
            PegSceneConfig,
            PegRewardWeights,
            PegRewardConfig,
            MjxPegTrainConfig,
        ]
        for cls in configs:
            obj = cls()
            assert obj is not None

    def test_removed_fields_stay_removed(self):
        # guards against accidental re-introduction of audit-removed fields;
        # each name was deleted in phase 4 (config dead-field cleanup). if
        # anyone adds one back by copy-paste, this test will flag it.
        w = RewardWeights()
        # opposition: removed — it paid side_ratio a second time on top of the
        # grasping term (same quantity, two payments).
        for name in ("action", "upward", "opposition"):
            assert not hasattr(w, name), f"RewardWeights.{name} should be removed"

        c = RewardConfig()
        # success_bonus: replaced by success_bonus_per_step — the one-shot +250
        # spike was clipped by VecNormalize early in training and needed a
        # latch to block yo-yo farming; the annuity needs neither.
        for name in ("hold_bonus", "success_bonus"):
            assert not hasattr(c, name), f"RewardConfig.{name} should be removed"

        pw = PegRewardWeights()
        # insertion_drive: removed 2026-07-14 — it paid gated downward peg
        # velocity with no matching charge on ascent, making a grip-and-bob
        # cycle net-positive (a non-potential shaping term, Ng '99); the place
        # keypoint term covers the same gradient without the loophole.
        for name in ("upward", "action_magnitude", "insertion_drive"):
            assert not hasattr(pw, name), f"PegRewardWeights.{name} should be removed"

        pc = PegRewardConfig()
        assert not hasattr(pc, "min_contacts_for_align"), (
            "PegRewardConfig.min_contacts_for_align should be removed"
        )
