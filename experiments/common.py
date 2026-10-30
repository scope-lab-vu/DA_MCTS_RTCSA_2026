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


MACHINE_LOG = REPO / "results" / "machine_log.csv"


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _ac_online(sysfs="/sys"):
    """'1' if ANY mains adapter is online, '0' if adapters exist but none is, '' if unknown."""
    seen = False
    for p in sorted((Path(sysfs) / "class/power_supply").glob("*")):
        if _read(p / "type") == "Mains":
            seen = True
            if _read(p / "online") == "1":
                return "1"
    return "0" if seen else ""


def _memory_state():
    """Available memory, swap in use, and cumulative memory-stall time (kernel PSI)."""
    out = {"mem_avail_mb": float("nan"), "swap_used_mb": float("nan"),
           "psi_some_us": float("nan"), "psi_full_us": float("nan")}
    try:
        mi = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            mi[k] = int(v.split()[0])
        out["mem_avail_mb"] = mi["MemAvailable"] / 1024.0
        out["swap_used_mb"] = (mi["SwapTotal"] - mi["SwapFree"]) / 1024.0
    except (OSError, KeyError, ValueError):
        pass
    try:
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            kind, rest = line.split(None, 1)
            for tok in rest.split():
                if tok.startswith("total="):
                    out[f"psi_{kind}_us"] = float(tok[len("total="):])
    except (OSError, ValueError):
        pass
    return out


def machine_preflight(core, sysfs="/sys"):
    """Check the machine is in a benchmarking state. Returns (problems, notes).

    problems should stop the sweep; notes are advice."""
    sysfs = Path(sysfs)
    problems, notes = [], []
    gov = _read(sysfs / f"devices/system/cpu/cpu{core}/cpufreq/scaling_governor")
    epp = _read(sysfs / f"devices/system/cpu/cpu{core}/cpufreq/energy_performance_preference")
    if gov is None:
        notes.append("CPU governor not readable (cannot verify it is 'performance')")
    elif gov == "performance":
        pass
    elif gov == "powersave" and epp == "performance":
        notes.append("governor reads 'powersave' but the energy preference is 'performance' "
                     "(equivalent on Intel HWP: accepted)")
    else:
        problems.append(f"CPU governor is '{gov}' (energy preference '{epp}'), expected "
                        "'performance'. Try: powerprofilesctl set performance")
    mem = _memory_state()
    if mem["mem_avail_mb"] == mem["mem_avail_mb"] and mem["mem_avail_mb"] < 3000:
        notes.append(f"only {mem['mem_avail_mb']:.0f} MB of memory available: close browsers "
                     "and other apps (memory pressure can stall the planner)")
    mains = [_read(p / "online") for p in sorted((sysfs / "class/power_supply").glob("*"))
             if _read(p / "type") == "Mains"]
    if mains and "1" not in mains:
        problems.append("running on battery (AC adapter not connected)")
    if _read(sysfs / "devices/system/cpu/intel_pstate/no_turbo") == "0":
        notes.append("turbo boost is ON: the clock can drop as the laptop heats up "
                     "(for stable timings consider no_turbo=1, see docs/methodology.md)")
    return problems, notes


def machine_snapshot(core):
    """Best-effort CPU frequency (planner core), hottest thermal zone, load, power state."""
    snap = {"freq_mhz": float("nan"), "temp_c": float("nan"), "load1": float("nan"),
            "governor": _read(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor") or "",
            "no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo") or "",
            "ac_online": ""}
    snap["ac_online"] = _ac_online()
    try:
        snap["freq_mhz"] = int(Path(
            f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq").read_text()) / 1000.0
    except (OSError, ValueError):
        pass
    temps = []
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(p.read_text()) / 1000.0)
        except (OSError, ValueError):
            pass
    if temps:
        snap["temp_c"] = max(temps)
    try:
        snap["load1"] = os.getloadavg()[0]
    except OSError:
        pass
    return snap


def log_machine(event, core, arm, load, seed, timing_only):
    """Append one row to results/machine_log.csv (evidence for throttling / drift)."""
    MACHINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not MACHINE_LOG.exists()
    snap = machine_snapshot(core)
    with open(MACHINE_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "event", "arm", "load", "seed", "variant",
                        "freq_mhz", "temp_c", "load1", "governor", "epp", "ac_online",
                        "no_turbo", "mem_avail_mb", "swap_used_mb", "psi_some_us", "psi_full_us"])
        mem = _memory_state()
        epp = _read(f"/sys/devices/system/cpu/cpu{core}/cpufreq/energy_performance_preference") or ""
        w.writerow([f"{time.time():.1f}", event, arm, load, seed,
                    "timing-only" if timing_only else "closed-loop",
                    snap["freq_mhz"], snap["temp_c"], snap["load1"],
                    snap["governor"], epp, snap["ac_online"], snap["no_turbo"],
                    f"{mem['mem_avail_mb']:.0f}", f"{mem['swap_used_mb']:.0f}",
                    mem["psi_some_us"], mem["psi_full_us"]])


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


def episode_command(cfg, seed, timing_only=False):
    st = cfg["study"]
    max_steps = st.get("timing_only_max_steps", st["max_steps"]) if timing_only else st["max_steps"]
    return ["taskset", "-c", str(st["planner_core"]), sys.executable, str(RUNNER),
            "--track", st["track"], "--mu", str(st["mu"]),
            "--time-budget", str(st["time_budget_s"]),
            "--num-trials", "1", "--base-seed", str(seed),
            "--max-steps", str(max_steps)]


def run_episode(cfg, arm, load, seed, n_iters=None, dry_run=False, timing_only=False):
    """Run ONE episode (one arm, one load, one seed). Returns the process return code.

    timing_only=True turns the late-command model off (tag gets the '-timing' suffix)."""
    if load not in cfg["loads"]:
        raise SystemExit(f"unknown load '{load}'; choose from {list(cfg['loads'])}")
    if arm == "fixed" and not n_iters:
        raise SystemExit("fixed arm needs N iterations: run `calibrate` or set fixed_n_iters")
    if not dry_run and shutil.which("taskset") is None:
        raise SystemExit("taskset not found: sudo dnf install util-linux")
    cmd = episode_command(cfg, seed, timing_only)
    env = episode_env(cfg, arm, load, n_iters, timing_only)
    with LoadInjector(load, cfg["loads"][load], seed, arm,
                      cfg["study"]["planner_core"], dry_run):
        if dry_run:
            tags = (f"DAMCTS_MODE={arm} DAMCTS_TAG={env['DAMCTS_TAG']} "
                    f"DAMCTS_LATE={env['DAMCTS_LATE']} DAMCTS_N_ITERS={env['DAMCTS_N_ITERS']}")
            print("   [dry-run]", tags, " ".join(cmd))
            return 0
        return subprocess.run(cmd, env=env, cwd=str(REPO)).returncode
