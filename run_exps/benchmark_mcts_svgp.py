"""Quick benchmark for MCTS SVGP calpha model.

Runs N steps on a given track/mu, dumps actions to a file for diff-comparison,
and reports per-step timing.  Reuses the setup logic from the experiment runner.

Usage:
    python run_exps/benchmark_mcts_svgp.py --max-steps 100 --output /tmp/mcts_baseline.txt
"""

import os as _os, sys as _sys; from pathlib import Path as _Path  # noqa: E702
_REPO = _Path(__file__).resolve().parent.parent
_os.chdir(_REPO); _sys.path.insert(0, str(_REPO))  # noqa: E702

import argparse
import logging
import random
import tempfile
import time

import numpy as np
import torch
import yaml

from src.api import F1TenthAPI
from src.track import Track

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("benchmark")

VEHICLE_PARAMS = _Path("params/vehicle_params.yaml")
BASE_CONFIG = _Path("params/config_mcts_dpw_svgp_calpha_action_prior_cpp.yaml")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_map_config_10489(track: str) -> _Path:
    """Return a temp map config with all paths pointing to mu=1.0489 files."""
    map_config_path = _Path(f"maps/config_{track}_map.yaml")
    with open(map_config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["map_path"] = f"./{track}_10489/{track}_map"
    cfg["centerline_path"] = f"./{track}_10489/{track}_centerline.csv"
    cfg["wpt_path"] = (
        f"./{track}_10489/"
        f"traj_race_cl_centerline_mintime_mue1.0489_{track}_map.csv"
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"bench_map_{track}_", dir=str(map_config_path.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return _Path(tmp.name)


def make_algo_config(lane_width_imag: float, lane_width_real: float) -> _Path:
    """Return a temp algo config with given lane widths."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["controller"]["lane_width"] = lane_width_imag
    cfg["controller"]["real_track_width"] = lane_width_real
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix="bench_algo_", dir=str(BASE_CONFIG.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return _Path(tmp.name)


def run_benchmark(track, mu, max_steps, seed, output_path, lane_width_imag=1.5,
                  lane_width_real=1.75):
    set_seed(seed)

    map_cfg_path = make_map_config_10489(track)
    algo_cfg_path = make_algo_config(lane_width_imag, lane_width_real)

    try:
        api = F1TenthAPI(VEHICLE_PARAMS, algo_cfg_path, map_cfg_path, mu_override=mu)
        waypoints = api.load_waypoints()

        centerline_path = api.map_config.centerline_path
        track_obj = Track(centerline_path)

        episode_logger = api.create_episode_logger()
        controller = api._controller

        actions = []
        step_times = []

        for step in range(max_steps):
            t0 = time.monotonic()
            state = api.get_state()
            cmd = api.compute_action(state, waypoints)
            done = api.apply_action(cmd)
            elapsed = time.monotonic() - t0
            step_times.append(elapsed)

            # Record: step, accel, steer_vel, v, x, y
            actions.append(
                f"{step},{cmd.acceleration:.10f},{cmd.steering_vel:.10f},"
                f"{state.v:.10f},{state.x:.10f},{state.y:.10f}"
            )

            if done:
                log.info("Episode ended at step %d", step)
                break

            if step % 20 == 0:
                log.info("Step %d/%d  v=%.2f  dt=%.4fs", step, max_steps, state.v, elapsed)

        # Write actions
        with open(output_path, "w") as f:
            f.write("step,acceleration,steering_vel,velocity,x,y\n")
            for line in actions:
                f.write(line + "\n")

        # Report timing
        ms = [t * 1000 for t in step_times]
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {track} mu={mu} ({len(actions)} steps)")
        print(f"{'='*60}")
        print(f"  Mean step time:   {np.mean(ms):.2f} ms")
        print(f"  Median step time: {np.median(ms):.2f} ms")
        print(f"  Std step time:    {np.std(ms):.2f} ms")
        print(f"  Min step time:    {np.min(ms):.2f} ms")
        print(f"  Max step time:    {np.max(ms):.2f} ms")
        print(f"  Total time:       {sum(step_times):.2f} s")
        print(f"  Actions saved to: {output_path}")
        print(f"{'='*60}\n")

    finally:
        _os.unlink(map_cfg_path)
        try:
            _os.unlink(algo_cfg_path)
        except OSError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="Sepang")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/mcts_actions.txt")
    parser.add_argument("--lane-width-imag", type=float, default=1.5)
    parser.add_argument("--lane-width-real", type=float, default=1.75)
    args = parser.parse_args()

    run_benchmark(args.track, args.mu, args.max_steps, args.seed, args.output,
                  args.lane_width_imag, args.lane_width_real)
