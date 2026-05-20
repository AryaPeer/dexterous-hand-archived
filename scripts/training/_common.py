from __future__ import annotations

from collections import defaultdict
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


def setup_sb3_logger(model, run_dir: Path) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(log_dir), ["stdout", "csv"]))
