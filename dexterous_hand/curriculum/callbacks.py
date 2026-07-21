import logging

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


def scale_stage_starts(
    stages: list[tuple],
    total_timesteps: int,
    reference_total_timesteps: int,
) -> list[tuple]:
    """Scale stage start steps from `reference_total_timesteps` to `total_timesteps`."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be > 0")
    if reference_total_timesteps <= 0:
        raise ValueError("reference_total_timesteps must be > 0")
    if not stages:
        return []

    scaled_stages: list[tuple] = []
    prev_start = 0
    for stage in stages:
        if len(stage) == 0:
            raise ValueError("stages cannot contain empty tuples")

        base_start = int(stage[0])
        scaled_start = int(round((base_start / reference_total_timesteps) * total_timesteps))
        scaled_start = min(max(scaled_start, 0), total_timesteps)
        scaled_start = max(scaled_start, prev_start)
        prev_start = scaled_start
        scaled_stages.append((scaled_start, *stage[1:]))

    scaled_stages[0] = (0, *scaled_stages[0][1:])
    return scaled_stages


class GraspCurriculumCallback(BaseCallback):
    def __init__(self, stages: list[tuple[int, float]], verbose: int = 0) -> None:
        super().__init__(verbose)
        self.stages = stages
        self._current_stage = 0

    def _apply(self, stage_idx: int) -> None:
        p_pre_grasped = float(self.stages[stage_idx][1])
        self.training_env.env_method("set_curriculum_params", p_pre_grasped)
        if self.verbose:
            logger.info(
                "[Curriculum] Grasp stage %d: p_pre_grasped=%.2f at step %d",
                stage_idx,
                p_pre_grasped,
                self.num_timesteps,
            )

    def _on_training_start(self) -> None:
        if not self.stages:
            return

        start_stage = 0
        for i, stage in enumerate(self.stages):
            if self.num_timesteps >= stage[0]:
                start_stage = i
        self._current_stage = start_stage
        self._apply(start_stage)

    def _on_step(self) -> bool:
        while (
            self._current_stage < len(self.stages) - 1
            and self.num_timesteps >= self.stages[self._current_stage + 1][0]
        ):
            self._current_stage += 1
            self._apply(self._current_stage)

        return True


class AssemblyCurriculumCallback(BaseCallback):
    def __init__(self, stages: list[tuple[int, float, float]], verbose: int = 0) -> None:
        super().__init__(verbose)
        self.stages = stages
        self._current_stage = 0

    def _on_training_start(self) -> None:
        if not self.stages:
            return

        start_stage = 0
        for i, stage in enumerate(self.stages):
            if self.num_timesteps >= stage[0]:
                start_stage = i
        self._current_stage = start_stage

        clearance = self.stages[start_stage][1]
        p_pre_grasped = float(self.stages[start_stage][2])
        self.training_env.env_method("set_curriculum_params", clearance, p_pre_grasped)

        if self.verbose:
            logger.info(
                "[Curriculum] Stage %d: clearance=%.1fmm, p_pre_grasped=%.2f at step %d",
                start_stage,
                clearance * 1000,
                p_pre_grasped,
                self.num_timesteps,
            )

    def _on_step(self) -> bool:
        while (
            self._current_stage < len(self.stages) - 1
            and self.num_timesteps >= self.stages[self._current_stage + 1][0]
        ):
            self._current_stage += 1
            clearance = self.stages[self._current_stage][1]
            p_pre_grasped = float(self.stages[self._current_stage][2])
            self.training_env.env_method("set_curriculum_params", clearance, p_pre_grasped)

            if self.verbose:
                logger.info(
                    "[Curriculum] Stage %d: clearance=%.1fmm, p_pre_grasped=%.2f at step %d",
                    self._current_stage,
                    clearance * 1000,
                    p_pre_grasped,
                    self.num_timesteps,
                )

        return True
