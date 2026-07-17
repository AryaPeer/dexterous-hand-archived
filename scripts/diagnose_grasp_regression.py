"""Diagnose a grasp training trajectory from its progress.csv."""
import csv
import sys

if len(sys.argv) != 2:
    sys.exit("usage: diagnose_grasp_regression.py <progress.csv>")

with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))


def col(name: str) -> list[float]:
    return [float(r[name]) if r[name] != "" else float("nan") for r in rows]


def safe_mean(xs: list[float]) -> float:
    clean = [x for x in xs if x == x]
    return sum(clean) / len(clean) if clean else float("nan")


def window_mean(xs: list[float], lo: int, hi: int) -> float:
    sub = [x for x in xs[lo:hi] if x == x]
    return sum(sub) / len(sub) if sub else float("nan")


timesteps = col("time/total_timesteps")
N = len(rows)

print(f"Iterations: {N}, timesteps: {timesteps[0]/1e6:.2f}M -> {timesteps[-1]/1e6:.2f}M\n")

print("=" * 100)
print("TASK METRICS over time (10 evenly-spaced windows)")
print("=" * 100)
print(f"{'iter':>5} {'M steps':>8} | {'obj_h':>7} {'succ_hold':>9} {'nfc':>6} {'fingdist':>9} {'obj_spd':>8} | {'ep_rew':>9} {'ep_len':>7}")
for k in range(10):
    lo = (k * N) // 10
    hi = ((k + 1) * N) // 10
    print(
        f"{rows[lo]['time/iterations']:>5} "
        f"{window_mean(timesteps, lo, hi)/1e6:>8.2f} | "
        f"{window_mean(col('train/metrics/object_height'), lo, hi):>7.4f} "
        f"{window_mean(col('train/metrics/success_hold_steps'), lo, hi):>9.3f} "
        f"{window_mean(col('train/metrics/num_finger_contacts'), lo, hi):>6.2f} "
        f"{window_mean(col('train/metrics/mean_fingertip_dist'), lo, hi):>9.4f} "
        f"{window_mean(col('train/metrics/object_speed'), lo, hi):>8.4f} | "
        f"{window_mean(col('rollout/ep_rew_mean'), lo, hi):>9.1f} "
        f"{window_mean(col('rollout/ep_len_mean'), lo, hi):>7.1f}"
    )

print()
print("=" * 100)
print("REWARD COMPONENTS over time")
print("=" * 100)
print(f"{'iter':>5} {'M steps':>8} | {'reach':>8} {'grasp':>8} {'grasp_q':>8} {'lift':>8} {'hold':>8} {'success':>8} {'drop':>8} {'idle':>8} {'act_pen':>8} {'total':>8}")
for k in range(10):
    lo = (k * N) // 10
    hi = ((k + 1) * N) // 10
    print(
        f"{rows[lo]['time/iterations']:>5} "
        f"{window_mean(timesteps, lo, hi)/1e6:>8.2f} | "
        f"{window_mean(col('train/reward/reaching'), lo, hi):>8.3f} "
        f"{window_mean(col('train/reward/grasping'), lo, hi):>8.3f} "
        f"{window_mean(col('train/reward/grasp_quality'), lo, hi):>8.3f} "
        f"{window_mean(col('train/reward/lifting'), lo, hi):>8.3f} "
        f"{window_mean(col('train/reward/holding'), lo, hi):>8.3f} "
        f"{window_mean(col('train/reward/success'), lo, hi):>8.3f} "
        f"{window_mean(col('train/reward/drop'), lo, hi):>8.3f} "
        f"{window_mean(col('train/reward/idle_penalty'), lo, hi):>8.3f} "
        f"{window_mean(col('train/reward/action_penalty'), lo, hi):>8.3f} "
        f"{window_mean(col('train/reward/total'), lo, hi):>8.3f}"
    )

print()
print("=" * 100)
print("POLICY HEALTH over time")
print("=" * 100)
print(f"{'iter':>5} {'M steps':>8} | {'std':>6} {'entropy':>9} {'approx_kl':>10} {'clip_frac':>10} {'expl_var':>9} {'val_loss':>10} {'pg_loss':>10}")
for k in range(10):
    lo = (k * N) // 10
    hi = ((k + 1) * N) // 10
    print(
        f"{rows[lo]['time/iterations']:>5} "
        f"{window_mean(timesteps, lo, hi)/1e6:>8.2f} | "
        f"{window_mean(col('train/std'), lo, hi):>6.3f} "
        f"{window_mean(col('train/entropy_loss'), lo, hi):>9.3f} "
        f"{window_mean(col('train/approx_kl'), lo, hi):>10.4f} "
        f"{window_mean(col('train/clip_fraction'), lo, hi):>10.4f} "
        f"{window_mean(col('train/explained_variance'), lo, hi):>9.3f} "
        f"{window_mean(col('train/value_loss'), lo, hi):>10.2f} "
        f"{window_mean(col('train/policy_gradient_loss'), lo, hi):>10.4f}"
    )

print()
print("=" * 100)
print("PEAK / TROUGH FINDER for key metrics")
print("=" * 100)

def peak_trough(name: str, key: str) -> None:
    vals = col(key)
    valid = [(i, v) for i, v in enumerate(vals) if v == v]
    if not valid:
        print(f"{name:>22}: no data")
        return
    peak_idx, peak_v = max(valid, key=lambda t: t[1])
    trough_idx, trough_v = min(valid, key=lambda t: t[1])
    final = next(v for _, v in reversed(valid))
    print(
        f"{name:>22}: peak={peak_v:.4f} at iter {peak_idx+1} ({timesteps[peak_idx]/1e6:.1f}M) | "
        f"trough={trough_v:.4f} at iter {trough_idx+1} ({timesteps[trough_idx]/1e6:.1f}M) | "
        f"final={final:.4f}"
    )

peak_trough("object_height", "train/metrics/object_height")
peak_trough("success_hold_steps", "train/metrics/success_hold_steps")
peak_trough("ep_rew_mean", "rollout/ep_rew_mean")
peak_trough("reward/lifting", "train/reward/lifting")
peak_trough("reward/holding", "train/reward/holding")
peak_trough("reward/success", "train/reward/success")
peak_trough("reward/grasp_quality", "train/reward/grasp_quality")
peak_trough("std", "train/std")
peak_trough("explained_variance", "train/explained_variance")
