import logging

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


def scale_stage_starts(
    stages: list[tuple],
    total_timesteps: int,
    reference_total_timesteps: int,
) -> list[tuple]:
    """Scale stage start steps from `reference_total_timesteps` to `total_timesteps`.

    Stages are `(start_step, *params)` tuples. Output is monotonic, clamped to
    [0, total_timesteps], and anchored at 0 for the first stage.
    """
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


class AssemblyCurriculumCallback(BaseCallback):
    def __init__(self, stages: list[tuple[int, float, float]], verbose: int = 0) -> None:
        super().__init__(verbose)
        self.stages = stages
        self._current_stage = 0

    def _on_training_start(self) -> None:
        if not self.stages:
            return

        # On a fresh run num_timesteps==0 -> stage 0. On a RESUME
        # (reset_num_timesteps=False) num_timesteps is already cumulative, so
        # jump straight to the correct stage rather than detouring through stage
        # 0 — which, for the peg env, would rebuild the model to the EASIEST
        # clearance and reset all envs, then fast-forward stage-by-stage with a
        # full XLA recompile at each boundary, and run the first step at the
        # wrong (stage-0) clearance/p_pre_grasped.
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
            # A clearance change rebuilds the model and resets every env
            # mid-rollout, so each env contributes one stale transition (the
            # pre-reset obs pairs with a post-reset next-obs) to GAE. That is
            # ~num_envs transitions per stage boundary, 4 boundaries per run
            # (~3k of 150M samples) — accepted; bridging it would need a
            # buffer-side episode cut.
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
