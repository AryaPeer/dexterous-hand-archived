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
        assert c.sim_timestep == 0.002
        assert c.frame_skip == 20

    def test_reward_weights(self):
        w = RewardWeights()
        assert w.reaching == 0.5
        assert w.grasping == 2.5
        assert w.lifting == 6.0
        assert w.holding == 6.0
        assert w.opposition == 1.0

    def test_reward_config(self):
        c = RewardConfig()
        assert isinstance(c.weights, RewardWeights)
        # 0.10 = real pick-up; guards against the round-11-13 erosion of the
        # lift bar (0.1 -> 0.07 -> 0.04 -> 0.012) that redefined the task as a
        # 12mm twitch instead of fixing the missing vertical arm DOF.
        assert c.lift_target == 0.10
        assert c.hold_height_smoothness_k == 50.0
        assert c.hold_velocity_smoothness_k == 20.0
        assert c.no_contact_idle_penalty == -0.08

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

    def test_peg_reward_config(self):
        c = PegRewardConfig()
        assert c.complete_bonus == 250.0
        assert c.force_threshold == 15.0
        assert c.idle_stage0_penalty == -0.3
        assert c.weights.opposition == 1.0
        assert c.weights.axis_in_grip == 1.0
        assert c.weights.insertion_drive == 3.0
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
        for name in ("action", "upward"):
            assert not hasattr(w, name), f"RewardWeights.{name} should be removed"

        c = RewardConfig()
        assert not hasattr(c, "hold_bonus"), "RewardConfig.hold_bonus should be removed"

        pw = PegRewardWeights()
        for name in ("upward", "action_magnitude"):
            assert not hasattr(pw, name), f"PegRewardWeights.{name} should be removed"

        pc = PegRewardConfig()
        assert not hasattr(pc, "min_contacts_for_align"), (
            "PegRewardConfig.min_contacts_for_align should be removed"
        )
