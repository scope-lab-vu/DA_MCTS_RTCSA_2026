"""Shared helpers for the contention study: config, load injection, episode launcher."""
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "run_exps" / "run_da_mcts_experiments.py"
CPU_LOAD = REPO / "experiments" / "cpu_load.py"
CONFIG = REPO / "experiments" / "study_config.yaml"
SUMMARY = REPO / "results" / "contention_summary.csv"
CALIB = REPO / "results" / "calibration.json"
LOAD_LOGS = REPO / "results" / "load_logs"


TIMING_SUFFIX = "-timing"   # tag suffix of the timing-only variant (late-command model off)


def tag_for(load, timing_only=False):
    """Tag stored in the results for a load condition and variant."""
    return f"{load}{TIMING_SUFFIX}" if timing_only else load


def load_config(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_n_iters(cfg):
    """Fixed-work iteration count: config value, else calibration.json, else None."""
    n = cfg["study"].get("fixed_n_iters")
    if n:
        return int(n)
    if CALIB.exists():
        return int(json.loads(CALIB.read_text())["suggested_n_iters"])
    return None


class LoadInjector:
    """Context manager: runs experiments/cpu_load.py in its own process group."""

    def __init__(self, name, spec, seed, arm, core, dry_run=False):
        self.name, self.spec, self.seed, self.arm = name, spec, seed, arm
        self.core = core   # the planner's core: all workers are pinned to it
        self.dry_run = dry_run
        self.proc = None

    def __enter__(self):
        if not self.spec.get("workers"):
            return self
        LOAD_LOGS.mkdir(parents=True, exist_ok=True)
        workers = ",".join([str(self.core)] * int(self.spec["workers"]))
        cmd = [sys.executable, str(CPU_LOAD), workers,
               self.spec.get("mode", "steady"), str(self.seed)]
        if self.dry_run:
            print("   [dry-run] load:", " ".join(cmd))
            return self
        log = open(LOAD_LOGS / f"load_{self.name}_s{self.seed}_{self.arm}.log", "w")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        time.sleep(1.0)   # let the workers spin up before the episode starts
        return self

    def __exit__(self, *exc):
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def already_done(arm, load, track, mu, tb_ms, seed):
    """True if results/contention_summary.csv already has a finished row for this run."""
    if not SUMMARY.exists():
        return False
    with open(SUMMARY, newline="") as f:
        for r in csv.DictReader(f):
            try:
                if (r["mode"] == arm and (r["tag"] or "none") == load and r["track"] == track
                        and abs(float(r["mu"]) - mu) < 1e-9
                        and abs(float(r["time_budget_ms"]) - tb_ms) < 1e-6
                        and int(r["seed"]) == seed and r["status"] == "OK"):
                    return True
            except (KeyError, ValueError):
                continue
    return False


def episode_env(cfg, arm, load, n_iters, timing_only=False):
    st = cfg["study"]
    env = os.environ.copy()
    late = st.get("late_model", True) and not timing_only
    env.update({
        "DAMCTS_MODE": arm,
        "DAMCTS_N_ITERS": str(n_iters or 1000),
        "DAMCTS_TAG": tag_for(load, timing_only),
        "DAMCTS_LATE": "1" if late else "0",
        "DAMCTS_WARMUP_DECISIONS": str(st.get("warmup_decisions", 40)),
        # one core => keep numeric libraries from spawning threads that fight each other
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    return env


def episode_command(cfg, seed):
    st = cfg["study"]
    return ["taskset", "-c", str(st["planner_core"]), sys.executable, str(RUNNER),
            "--track", st["track"], "--mu", str(st["mu"]),
            "--time-budget", str(st["time_budget_s"]),
            "--num-trials", "1", "--base-seed", str(seed),
            "--max-steps", str(st["max_steps"])]


def run_episode(cfg, arm, load, seed, n_iters=None, dry_run=False, timing_only=False):
    """Run ONE episode (one arm, one load, one seed). Returns the process return code.

    timing_only=True turns the late-command model off (tag gets the '-timing' suffix)."""
    if load not in cfg["loads"]:
        raise SystemExit(f"unknown load '{load}'; choose from {list(cfg['loads'])}")
    if arm == "fixed" and not n_iters:
        raise SystemExit("fixed arm needs N iterations: run `calibrate` or set fixed_n_iters")
    if not dry_run and shutil.which("taskset") is None:
        raise SystemExit("taskset not found: sudo dnf install util-linux")
    cmd = episode_command(cfg, seed)
    env = episode_env(cfg, arm, load, n_iters, timing_only)
    with LoadInjector(load, cfg["loads"][load], seed, arm,
                      cfg["study"]["planner_core"], dry_run):
        if dry_run:
            tags = (f"DAMCTS_MODE={arm} DAMCTS_TAG={env['DAMCTS_TAG']} "
                    f"DAMCTS_LATE={env['DAMCTS_LATE']} DAMCTS_N_ITERS={env['DAMCTS_N_ITERS']}")
            print("   [dry-run]", tags, " ".join(cmd))
            return 0
        return subprocess.run(cmd, env=env, cwd=str(REPO)).returncode
