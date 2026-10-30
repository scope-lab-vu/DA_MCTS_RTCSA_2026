"""Orchestrator for the CPU-contention study.

    python experiments/cpu_contention_experiments.py calibrate   # 1 idle episode -> N for the fixed arm
    python experiments/cpu_contention_experiments.py run         # the whole sweep (resumable)
    python experiments/cpu_contention_experiments.py status      # how much is done

Useful flags for `run`:
    --dry-run            print what would run, run nothing
    --timing-only        variant with the late-command model OFF (clean hit-rate statistics)
    --seeds 3            use only the first 3 seeds (pilot)
    --loads none heavy   subset of load conditions
    --arms fixed da      subset of planner arms

The sweep skips any (arm, load, seed) already present in results/contention_summary.csv,
so you can stop it (Ctrl-C) and restart it later.
"""
import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def preflight(cfg, args):
    """Refuse to start on a machine that is not in a benchmarking state (override: --force)."""
    problems, notes = common.machine_preflight(cfg["study"]["planner_core"])
    for n in notes:
        print("   note:", n)
    if problems and not args.dry_run:
        if args.force:
            print("   WARNING (ignored because of --force):", "; ".join(problems))
            return True
        print("MACHINE NOT READY:")
        for p in problems:
            print("  -", p)
        print("Fix these (docs/methodology.md, section 11) or override with --force.")
        return False
    return True


def cmd_calibrate(cfg, args):
    if not preflight(cfg, args):
        return 1
    st = cfg["study"]
    seed = st["seeds"][0]
    print(f"Calibration: 1 idle episode of the 'da' arm, planner pinned to core {st['planner_core']}")
    rc = common.run_episode(cfg, "da", "none", seed, None, args.dry_run)
    if args.dry_run:
        return 0
    # NOTE: this idle episode has exactly the configuration of study cell
    # (arm=da, load=none, seed=seeds[0]), so it also counts as that data point and
    # `run` will skip that cell. Re-calibrate after changing time_budget_s (new files).
    pattern = str(common.REPO / "results" / "timing" /
                  f"timing_da_none_{st['track']}_mu*_tb*ms_t1_s{seed}.csv")
    files = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    if not files:
        print("No timing file found. The episode probably failed (return code", rc, ").")
        return 1
    d = np.genfromtxt(files[-1], delimiter=",", names=True)
    m = (d["decision"] > st.get("warmup_decisions", 40)) & (d["n_iters"] >= 0)
    if m.sum() < 50:
        print(f"Only {int(m.sum())} usable decisions (episode ended early?). Look at the run output.")
        return 1
    plan_ms = d["plan_time_s"][m] * 1000.0
    iters = d["n_iters"][m]
    period_ms = st["period_s"] * 1000.0
    hit = float((plan_ms <= period_ms).mean())
    p = np.percentile(plan_ms, [50, 95, 99])
    it = np.percentile(iters, [10, 50, 90])
    overhead = p[0] - st["time_budget_s"] * 1000.0
    suggested = max(50, int(it[0] // 50 * 50))
    print(f"\nFile: {files[-1]}")
    print(f"Decisions after warm-up      : {int(m.sum())}")
    print(f"Deadline hit rate (<= {period_ms:.0f} ms) : {hit:.3f}")
    print(f"Plan time ms  p50/p95/p99    : {p[0]:.1f} / {p[1]:.1f} / {p[2]:.1f}")
    print(f"Overhead outside the budget  : about {overhead:.1f} ms (p50 minus budget)")
    print(f"Iterations p10/p50/p90       : {it[0]:.0f} / {it[1]:.0f} / {it[2]:.0f}")
    print(f"Suggested N for fixed arm    : {suggested}  (10th percentile, rounded down to 50)")
    if hit < 0.97:
        want = st["time_budget_s"] * 1000.0 - max(0.0, p[1] - period_ms) - 1.0
        print(f"\nHit rate is below 97% on an IDLE machine. Consider time_budget_s about "
              f"{max(0.005, want / 1000.0):.3f} and calibrate again.")
    common.CALIB.parent.mkdir(parents=True, exist_ok=True)
    common.CALIB.write_text(json.dumps({
        "suggested_n_iters": suggested, "hit_rate_idle": hit,
        "plan_ms_p50_p95_p99": p.tolist(), "iters_p10_p50_p90": it.tolist(),
        "time_budget_s": st["time_budget_s"], "source": files[-1],
    }, indent=2))
    print(f"\nSaved {common.CALIB}")
    return 0


def build_tasks(cfg, args):
    st = cfg["study"]
    seeds = st["seeds"][: args.seeds] if args.seeds else st["seeds"]
    loads = args.loads or list(cfg["loads"])
    arms = args.arms or st["arms"]
    tasks = []
    for seed in seeds:
        block = [(arm, load, seed) for load in loads for arm in arms]
        random.Random(seed * 1000).shuffle(block)   # whole seed block shuffled, reproducibly:
        tasks += block                              # no load or arm always runs first/last
    return tasks


def cmd_run(cfg, args):
    if not preflight(cfg, args):
        return 1
    st = cfg["study"]
    tb_ms = st["time_budget_s"] * 1000.0
    n_iters = common.resolve_n_iters(cfg)
    tasks = build_tasks(cfg, args)
    if any(a == "fixed" for a, _, _ in tasks) and not n_iters:
        if not args.dry_run:
            print("The fixed arm needs N iterations. Run `calibrate` first (or set fixed_n_iters).")
            return 1
        n_iters = 900   # placeholder, dry run only
        print("(dry run: using placeholder N = 900 for the fixed arm)")
    todo = [t for t in tasks
            if not common.already_done(t[0], common.tag_for(t[1], args.timing_only),
                                       st["track"], st["mu"], tb_ms, t[2])]
    print(f"Planned episodes: {len(tasks)}   already done: {len(tasks) - len(todo)}   "
          f"to run: {len(todo)}   (about {len(todo) * 4 / 60:.1f} h at ~4 min each)")
    print(f"Track {st['track']}  mu {st['mu']}  budget {tb_ms:.0f} ms  fixed N {n_iters}  "
          f"planner core {st['planner_core']}  variant: "
          f"{'TIMING-ONLY (late model off)' if args.timing_only else 'closed-loop (late model on)'}\n")
    failed = []
    for i, (arm, load, seed) in enumerate(todo, 1):
        print(f">>> [{i}/{len(todo)}] arm={arm} load={load} seed={seed}  ({time.strftime('%H:%M:%S')})")
        if not args.dry_run:
            common.log_machine("before", st["planner_core"], arm, load, seed, args.timing_only)
        rc = common.run_episode(cfg, arm, load, seed, n_iters, args.dry_run, args.timing_only)
        if not args.dry_run:
            common.log_machine("after", st["planner_core"], arm, load, seed, args.timing_only)
            if st.get("cooldown_s", 0) > 0:
                time.sleep(st["cooldown_s"])   # let the CPU shed heat between episodes
        if rc != 0:
            failed.append((arm, load, seed))
            print(f"    RUN FAILED (return code {rc})")
    if failed:
        print("\nFailed runs (rerun `run` to retry them):", failed)
    print("\nDone. Next: python analysis/summarize_results.py && python analysis/generate_plots.py")
    return 0


def cmd_status(cfg, args):
    st = cfg["study"]
    tb_ms = st["time_budget_s"] * 1000.0
    print("variant:", "TIMING-ONLY" if args.timing_only else "closed-loop")
    print(f"{'load':<8}" + "".join(f"{a:>10}" for a in st["arms"]) + "   (done / planned seeds)")
    for load in cfg["loads"]:
        row = f"{load:<8}"
        for arm in st["arms"]:
            n = sum(common.already_done(arm, common.tag_for(load, args.timing_only),
                                        st["track"], st["mu"], tb_ms, s)
                    for s in st["seeds"])
            row += f"{n:>6}/{len(st['seeds']):<3}"
        print(row)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["calibrate", "run", "status"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="start even if the machine preflight reports problems")
    ap.add_argument("--timing-only", action="store_true",
                    help="late-command model off; results tagged '<load>-timing'")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--loads", nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    args = ap.parse_args()
    cfg = common.load_config()
    sys.exit({"calibrate": cmd_calibrate, "run": cmd_run, "status": cmd_status}[args.command](cfg, args))


if __name__ == "__main__":
    main()
