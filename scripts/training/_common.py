from __future__ import annotations

from collections import defaultdict, deque
import contextlib
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure


class RewardInfoLoggerCallback(BaseCallback):
    """Aggregate per-step reward/metric infos and log their means at rollout end."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._buf: dict[str, list[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if not (k.startswith("reward/") or k.startswith("metrics/")):
                    continue
                try:
                    self._buf[k].append(float(v))
                except (TypeError, ValueError):
                    continue
        return True

    def _on_rollout_end(self) -> None:
        for k, vals in self._buf.items():
            if vals:
                self.logger.record(f"train/{k}", float(np.mean(vals)))
        self._buf.clear()


# A single gated metric: (info_key, floor, baseline_5m, why).
#   info_key   - raw env-info key, e.g. "metrics/axis_align" / "reward/complete"
#   floor      - the recent mean must stay >= this or the gate STOPS the run
#   baseline_5m- the value observed in the 2026-06-01 5M sanity (for the report)
#   why        - one-line human reason printed in the diagnostic
Check = tuple[str, float, float, str]
# A milestone: (timestep, [Check, ...], label).
Milestone = tuple[int, list[Check], str]


class MilestoneGateCallback(BaseCallback):
    """Compute-saver gate. At each milestone timestep, print a diagnostic of the
    gated task metrics (recent rollout means) against floors derived from the
    round-16-fix 5M sanity, and STOP training early if any metric is clearly
    below floor — a regression/collapse, or a progress metric gone flat.

    This is a *diagnostic* stop, not a silent kill-switch: it prints each
    metric, its 5M baseline, the floor and a one-line reason, so the failure
    mode is identifiable straight from the log. ``CheckpointCallback`` saves
    every ~500k steps, so a stop here always leaves a resumable checkpoint — if
    you judge the stop premature, resume with ``resume-{peg,grasp}-mjx``;
    otherwise fix the cause and restart. Disable entirely with ``--no-gate``.

    The gate only reads keys present in the env ``info`` dicts ("metrics/..."
    and "reward/..."). Unknown keys are reported as SKIP and never fail the
    gate, so a typo or a renamed metric can't kill a healthy run.
    """

    def __init__(
        self, milestones: list[Milestone], window_rollouts: int = 15, verbose: int = 1
    ) -> None:
        super().__init__(verbose)
        self._milestones = sorted(milestones, key=lambda m: m[0])
        self._idx = 0
        self._window = window_rollouts
        self._roll: dict[str, list[float]] = defaultdict(list)
        self._recent: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_rollouts))
        self._n_rollouts = 0
        self._stop = False

    def _on_training_start(self) -> None:
        # On a resume (num_timesteps already cumulative) skip milestones that
        # are already behind us, so a deliberately-resumed run is not
        # insta-stopped on its first rollout.
        while (
            self._idx < len(self._milestones)
            and self.num_timesteps >= self._milestones[self._idx][0]
        ):
            self._idx += 1

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if k.startswith("metrics/") or k.startswith("reward/"):
                    try:
                        self._roll[k].append(float(v))
                    except (TypeError, ValueError):
                        continue
        return not self._stop

    def _on_rollout_end(self) -> None:
        for k, vals in self._roll.items():
            if vals:
                self._recent[k].append(float(np.mean(vals)))
        self._roll.clear()
        self._n_rollouts += 1
        while (
            self._idx < len(self._milestones)
            and self.num_timesteps >= self._milestones[self._idx][0]
        ):
            _, checks, label = self._milestones[self._idx]
            try:
                self._evaluate(checks, label)
            except Exception as exc:  # a gate bug must never kill a healthy run
                print(f"[MilestoneGate] WARNING: gate eval errored ({exc!r}); continuing.")
            self._idx += 1

    def _recent_mean(self, key: str) -> float | None:
        vals = self._recent.get(key)
        return float(np.mean(vals)) if vals else None

    def evaluate_checks(
        self, checks: list[Check]
    ) -> tuple[list[tuple], list[Check]]:
        """Pure decision step (unit-testable): returns (rows, failures).

        ``rows`` is one (tag, key, value-or-None, floor, baseline, why) per
        check for the report; ``failures`` is the subset whose recent mean is
        below floor. A check whose metric was never observed is reported SKIP
        and is NOT a failure.
        """
        rows: list[tuple] = []
        failures: list[Check] = []
        for key, floor, base, why in checks:
            cur = self._recent_mean(key)
            if cur is None:
                rows.append(("SKIP", key, None, floor, base, why))
                continue
            if cur >= floor:
                rows.append(("OK", key, cur, floor, base, why))
            else:
                rows.append(("FAIL", key, cur, floor, base, why))
                failures.append((key, floor, base, why))
        return rows, failures

    def _evaluate(self, checks: list[Check], label: str) -> None:
        rows, failures = self.evaluate_checks(checks)
        n = min(self._n_rollouts, self._window)
        lines = [
            f"\n===== MILESTONE GATE @ {self.num_timesteps:,} steps — {label} =====",
            f"  recent mean over last {n} rollout(s):",
        ]
        for tag, key, cur, floor, base, why in rows:
            if cur is None:
                lines.append(f"  [SKIP] {key:32s}  metric not seen — not gated")
            else:
                lines.append(
                    f"  [{tag:<4}] {key:32s} = {cur:9.4f}   "
                    f"5M={base:<8.4g} floor>={floor:<8.4g} — {why}"
                )
        if failures:
            lines.append("")
            lines.append(f"  VERDICT: STOP — {len(failures)} metric(s) below floor:")
            for key, floor, _base, why in failures:
                lines.append(f"    - {key} < {floor}  ({why})")
            lines.append("  A ~500k-cadence checkpoint exists under runs/<run>/checkpoints/.")
            lines.append(
                "  Premature? resume with resume-{peg,grasp}-mjx. "
                "Otherwise fix the cause and restart."
            )
            self._stop = True
        else:
            lines.append("")
            lines.append("  VERDICT: PASS — all gated metrics above floor; continuing.")
        print("\n".join(lines), flush=True)
        # logging is best-effort; the stop decision is already made above
        with contextlib.suppress(Exception):
            self.logger.record("gate/passed", 0.0 if failures else 1.0)


def setup_sb3_logger(model, run_dir: Path) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(log_dir), ["stdout", "csv"]))
