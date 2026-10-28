"""Shared helpers for the analysis scripts (numpy + pandas only)."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results" / "contention_summary.csv"
TIMING_DIR = REPO / "results" / "timing"
OUT = REPO / "results" / "analysis"

TIMING_SUFFIX = "-timing"   # tag suffix of the timing-only variant (late model off)
ARM_ORDER = ["fixed", "anytime", "da"]
ARM_LABEL = {"fixed": "Fixed work (baseline)",
             "anytime": "Anytime, uniform weights",
             "da": "DA-MCTS (deadline-aware)"}
ARM_COLOR = {"fixed": "tab:red", "anytime": "tab:orange", "da": "tab:blue"}
LOAD_ORDER = ["none", "light", "heavy", "bursty"]


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n (honest for small n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(x, n_boot=10000, seed=0):
    """Percentile bootstrap 95% CI of the mean. Returns (lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


NUMERIC = ["mu", "time_budget_ms", "trial", "seed", "laps", "steps", "progress_pct",
           "best_lap_s", "mean_lap_s", "deadline_hit_rate", "plan_mean_ms", "plan_median_ms",
           "plan_std_ms", "plan_p95_ms", "plan_p99_ms", "plan_max_ms", "severe_miss_rate",
           "mean_iters", "n_decisions", "late_steps_total", "steer_rate_change", "accel_change"]


def out_dir(timing_only=False):
    return OUT / "timing_only" if timing_only else OUT


def load_summary(path=SUMMARY, timing_only=False):
    """Read the per-episode summary; keep finished episodes, one row per (arm, load, seed).

    timing_only=False -> closed-loop episodes (late model on);
    timing_only=True  -> timing-only episodes (tag '<load>-timing'); the suffix is stripped."""
    if not Path(path).exists():
        raise SystemExit(f"{path} not found. Run the experiments first.")
    df = pd.read_csv(path)
    df = df[df["status"] == "OK"].copy()
    df["tag"] = df["tag"].fillna("none").astype(str)
    is_t = df["tag"].str.endswith(TIMING_SUFFIX)
    df = df[is_t].copy() if timing_only else df[~is_t].copy()
    if timing_only:
        df["tag"] = df["tag"].str[: -len(TIMING_SUFFIX)]
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["collision"] = df["collision"].astype(str).str.lower().eq("true")
    df["completed"] = df["reason"].astype(str).eq("done")   # finished all laps, no crash
    df = df.drop_duplicates(subset=["mode", "tag", "track", "mu", "time_budget_ms", "seed"],
                            keep="last")
    return df


def build_condition_table(df):
    """One row per (arm, load) with the study's metrics."""
    rows = []
    for arm in ARM_ORDER:
        for load in LOAD_ORDER:
            g = df[(df["mode"] == arm) & (df["tag"] == load)]
            n = len(g)
            if n == 0:
                continue
            ncol = int(g["collision"].sum())
            ncomp = int(g["completed"].sum())
            c_lo, c_hi = wilson(ncol, n)
            k_lo, k_hi = wilson(ncomp, n)
            h_lo, h_hi = bootstrap_mean_ci(g["deadline_hit_rate"])
            done = g[g["completed"]]
            crashed = g[g["collision"]]
            rows.append({
                "arm": arm, "load": load, "episodes": n,
                "completion_rate": ncomp / n, "completion_lo": k_lo, "completion_hi": k_hi,
                "collision_rate": ncol / n, "collision_lo": c_lo, "collision_hi": c_hi,
                "progress_mean": g["progress_pct"].mean(), "progress_sd": g["progress_pct"].std(),
                "progress_median": g["progress_pct"].median(),
                "hit_mean": g["deadline_hit_rate"].mean(), "hit_sd": g["deadline_hit_rate"].std(),
                "hit_lo": h_lo, "hit_hi": h_hi,
                "plan_median_ms": g["plan_median_ms"].mean(),
                "plan_mean_ms": g["plan_mean_ms"].mean(),
                "plan_std_ms": g["plan_std_ms"].mean(),
                "plan_p99_ms": g["plan_p99_ms"].mean(),
                "plan_max_ms_worst": g["plan_max_ms"].max(),
                "severe_miss_rate": g["severe_miss_rate"].mean(),
                "mean_iters": g["mean_iters"].mean(),
                "decisions_mean": g["n_decisions"].mean() if "n_decisions" in g else float("nan"),
                "mean_lap_s": done["mean_lap_s"].mean() if len(done) else float("nan"),
                "best_lap_s": done["best_lap_s"].mean() if len(done) else float("nan"),
                "steps_before_crash": crashed["steps"].mean() if len(crashed) else float("nan"),
                "steer_rate_change": g["steer_rate_change"].mean(),
                "accel_change": g["accel_change"].mean(),
            })
    return pd.DataFrame(rows)


def paired_differences(df, a, b, metrics, n_boot=10000):
    """Per load: mean of (arm a - arm b) over seeds present in both, with bootstrap 95% CI."""
    out = []
    for load in LOAD_ORDER:
        ga = df[(df["mode"] == a) & (df["tag"] == load)].set_index("seed")
        gb = df[(df["mode"] == b) & (df["tag"] == load)].set_index("seed")
        seeds = ga.index.intersection(gb.index)
        if len(seeds) == 0:
            continue
        for m in metrics:
            x = ga.loc[seeds, m].astype(float).values
            y = gb.loc[seeds, m].astype(float).values
            d = x - y
            d = d[np.isfinite(d)]
            if len(d) == 0:
                continue
            lo, hi = bootstrap_mean_ci(d, n_boot=n_boot)
            out.append({"comparison": f"{a} - {b}", "load": load, "metric": m,
                        "pairs": len(d), "mean_diff": float(d.mean()),
                        "ci_lo": lo, "ci_hi": hi,
                        "ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(out)
