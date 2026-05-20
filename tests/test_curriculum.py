from unittest.mock import MagicMock

from dexterous_hand.curriculum.callbacks import (
    AssemblyCurriculumCallback,
    ReorientCurriculumCallback,
    scale_stage_starts,
)


def _setup_callback(cb: ReorientCurriculumCallback | AssemblyCurriculumCallback) -> MagicMock:

    # BaseCallback.training_env is a @property that resolves to
    # self.model.get_env(), so mocking cb.model.get_env is sufficient — no
    # need (or possibility) to set cb.training_env directly.
    mock_env = MagicMock()
    cb.locals = {}
    cb.globals = {}
    cb.model = MagicMock()
    cb.model.get_env.return_value = mock_env
    return mock_env

class TestReorientCurriculumCallback:
    def test_applies_stage_zero_on_training_start(self) -> None:
        stages = [(0, 0.5), (20_000_000, 1.57)]
        cb = ReorientCurriculumCallback(stages)
        mock_env = _setup_callback(cb)
        cb._on_training_start()
        mock_env.env_method.assert_called_once_with("set_curriculum_stage", 0.5)

    def test_no_transition_at_start(self) -> None:
        stages = [(0, 0.5), (20_000_000, 1.57), (60_000_000, 3.14)]
        cb = ReorientCurriculumCallback(stages)
        mock_env = _setup_callback(cb)
        cb.num_timesteps = 0
        cb._on_step()
        mock_env.env_method.assert_not_called()

    def test_transition_at_threshold(self) -> None:
        stages = [(0, 0.5), (100, 1.57)]
        cb = ReorientCurriculumCallback(stages)
        mock_env = _setup_callback(cb)
        cb.num_timesteps = 100
        cb._on_step()
        mock_env.env_method.assert_called_once_with("set_curriculum_stage", 1.57)

    def test_multiple_transitions(self) -> None:
        stages = [(0, 0.5), (100, 1.57), (200, 3.14)]
        cb = ReorientCurriculumCallback(stages)
        _setup_callback(cb)
        cb.num_timesteps = 250
        cb._on_step()
        assert cb._current_stage == 2

class TestAssemblyCurriculumCallback:
    def test_applies_stage_zero_on_training_start(self) -> None:
        stages = [(0, 0.004, True), (25_000_000, 0.004, False)]
        cb = AssemblyCurriculumCallback(stages)
        mock_env = _setup_callback(cb)
        cb._on_training_start()
        mock_env.env_method.assert_called_once_with("set_curriculum_params", 0.004, True)

    def test_no_transition_at_start(self) -> None:
        stages = [(0, 0.004, True), (25_000_000, 0.004, False)]
        cb = AssemblyCurriculumCallback(stages)
        mock_env = _setup_callback(cb)
        cb.num_timesteps = 0
        cb._on_step()
        mock_env.env_method.assert_not_called()

    def test_transition(self) -> None:
        stages = [(0, 0.004, True), (100, 0.002, False)]
        cb = AssemblyCurriculumCallback(stages)
        mock_env = _setup_callback(cb)
        cb.num_timesteps = 100
        cb._on_step()
        mock_env.env_method.assert_called_once_with("set_curriculum_params", 0.002, False)

    def test_returns_true(self) -> None:
        stages = [(0, 0.004, True)]
        cb = AssemblyCurriculumCallback(stages)
        _setup_callback(cb)
        cb.num_timesteps = 0
        assert cb._on_step() is True

class TestScaleStageStarts:
    def test_scales_reorient_stages_with_total_timesteps(self) -> None:
        stages = [
            (0, 0.5),
            (20_000_000, 1.57),
            (60_000_000, 3.14),
            (120_000_000, 3.14159),
        ]
        scaled = scale_stage_starts(
            stages=stages,
            total_timesteps=120_000_000,
            reference_total_timesteps=400_000_000,
        )
        assert [stage[0] for stage in scaled] == [0, 6_000_000, 18_000_000, 36_000_000]

    def test_scales_assembly_stages_and_preserves_payload(self) -> None:
        stages = [
            (0, 0.004, True),
            (25_000_000, 0.004, False),
            (50_000_000, 0.002, False),
            (75_000_000, 0.001, False),
        ]
        scaled = scale_stage_starts(
            stages=stages,
            total_timesteps=200_000_000,
            reference_total_timesteps=100_000_000,
        )
        assert [stage[0] for stage in scaled] == [0, 50_000_000, 100_000_000, 150_000_000]
        assert scaled[2][1:] == (0.002, False)

    def test_raises_on_non_positive_total_timesteps(self) -> None:
        try:
            scale_stage_starts([(0, 0.5)], total_timesteps=0, reference_total_timesteps=400_000_000)
            raise AssertionError("Expected ValueError for total_timesteps <= 0")
        except ValueError as err:
            assert "total_timesteps" in str(err)
