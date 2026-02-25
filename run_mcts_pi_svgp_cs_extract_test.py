"""
Test run: MCTS DPW + PI-SVGP + Cornering Stiffness + best-path extraction.

Runs a single episode on YasMarina with mu=1.0 and logs the best MCTS path
(Aa_action_list, Asv_action_list) at each step.

Usage:
    python run_mcts_pi_svgp_cs_extract_test.py                  # no render
    python run_mcts_pi_svgp_cs_extract_test.py --render          # with visualisation
    python run_mcts_pi_svgp_cs_extract_test.py --max-steps 5000  # limit episode
"""

import argparse
import logging
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.api import F1TenthAPI
from src.track import Track

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("mcts_cs_extract_test")

TRACK = "YasMarina"
DEFAULT_MU = 1.0

BASE_CONFIG = Path("params/config_mcts_dpw_pi_svgp_cpp_cs_extract.yaml")
VEHICLE_PARAMS = Path("params/vehicle_params.yaml")
RESULTS_DIR = Path("results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_map_config_all_10489_files(map_config_path: Path, track: str) -> Path:
    """Return a temp map config YAML with ALL file paths hardcoded to mu=1.0489."""
    with open(map_config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["map_path"] = f"./{track}_10489/{track}_map"
    cfg["centerline_path"] = f"./{track}_10489/{track}_centerline.csv"
    cfg["wpt_path"] = (
        f"./{track}_10489/"
        f"traj_race_cl_centerline_mintime_mue1.0489_{track}_map.csv"
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        prefix=f"map_cs_{track}_",
        dir=str(map_config_path.parent),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def main():
    parser = argparse.ArgumentParser(
        description="Test MCTS + PI-SVGP + CS + best-path extraction"
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--mu", type=float, default=DEFAULT_MU, help="Friction coefficient")
    args = parser.parse_args()

    MU = args.mu

    set_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    map_config_path = Path(f"maps/config_{TRACK}_map.yaml")
    if not map_config_path.exists():
        log.error("Map config not found: %s", map_config_path)
        return

    fixed_map_config_path = make_map_config_all_10489_files(map_config_path, TRACK)

    try:
        api = F1TenthAPI(VEHICLE_PARAMS, BASE_CONFIG, fixed_map_config_path, mu_override=MU)
        waypoints = api.load_waypoints()

        centerline_path = api.map_config.centerline_path
        track_obj = Track(centerline_path)

        if args.render:
            api.setup_rendering(waypoints)

        episode_logger = api.create_episode_logger()
        dt_s = api.algo_config.simulation.dt_s

        # Get direct access to the controller for CS + best-path introspection.
        controller = api._controller

        step = 0
        reason = "max_steps"
        step_times = []
        prev_progress = None
        accumulated_progress = 0.0
        prev_laps = 0
        lap_step_start = 0
        lap_times_list: list = []

        try:
            while step < args.max_steps:
                t0 = time.monotonic()

                state = api.get_state()
                cmd = api.compute_action(state, waypoints)

                # Extract best-path action lists from the controller.
                accel_path = controller.get_best_path_accel()
                steer_path = controller.get_best_path_steer()
                aa_str = ";".join(f"{a:.6f}" for a in accel_path)
                asv_str = ";".join(f"{s:.6f}" for s in steer_path)

                # Extract cornering stiffness values.
                cs_raw = controller.get_cornering_stiffness()
                cs_smooth = controller.get_smoothed_calpha()

                episode_logger.log_step(
                    state, cmd,
                    float("nan"), float("nan"), float("nan"),
                    aa_action_list=aa_str,
                    asv_action_list=asv_str,
                    c_alpha=cs_raw,
                    c_alpha_smooth=cs_smooth,
                )
                done = api.apply_action(cmd)

                if args.render:
                    api.render()

                elapsed = time.monotonic() - t0
                step_times.append(elapsed)
                step += 1

                # Track progress.
                pos = np.array([state.x, state.y])
                _, _, progress = track_obj.get_nearest_point(pos)
                if prev_progress is None:
                    prev_progress = progress

                delta_p = progress - prev_progress
                if delta_p < -track_obj.total_length * 0.5:
                    delta_p += track_obj.total_length
                elif delta_p > track_obj.total_length * 0.5:
                    delta_p -= track_obj.total_length
                accumulated_progress += delta_p
                prev_progress = progress

                laps_completed = int(accumulated_progress / track_obj.total_length)

                if laps_completed > prev_laps:
                    for _ in range(laps_completed - prev_laps):
                        lap_time_s = (step - lap_step_start) * dt_s
                        lap_times_list.append(round(lap_time_s, 3))
                        lap_step_start = step
                    prev_laps = laps_completed

                if step % 100 == 0:
                    pct = (accumulated_progress / track_obj.total_length) * 100
                    cs = controller.get_cornering_stiffness()
                    c_alpha_smooth = controller.get_smoothed_calpha()
                    n_actions = len(accel_path)
                    log.info(
                        "[%s mu=%.2f] step=%d  v=%.2f  progress=%.1f%%  "
                        "C_alpha=%.1f (smooth=%.1f)  path_len=%d  dt=%.4fs",
                        TRACK, MU, step, state.v, pct, cs, c_alpha_smooth,
                        n_actions, elapsed,
                    )

                if done:
                    reason = "collision" if api.collision else "done"
                    log.info(
                        "[%s mu=%.2f] Finished at step %d (%s)",
                        TRACK, MU, step, reason,
                    )
                    break

                sleep_time = dt_s - elapsed
                if sleep_time > 0 and args.render:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            reason = "interrupted"
            log.info("[%s mu=%.2f] Interrupted at step %d", TRACK, MU, step)

        laps = int(accumulated_progress / track_obj.total_length)
        mu_id = str(MU).replace(".", "")
        output_dir = str(RESULTS_DIR / f"mcts_pi_svgp_cs_extract_{TRACK}_{mu_id}")

        extra_meta = {
            "seed": args.seed,
            "lap_times_s": lap_times_list,
            "collision": reason == "collision",
        }
        csv_path = api.save_episode(
            episode_logger,
            output_dir=output_dir,
            reason=reason,
            config_name=f"{BASE_CONFIG.stem}_s{args.seed}",
            extra_metadata=extra_meta,
        )

        avg_dt = np.mean(step_times) if step_times else 0.0
        progress_pct = (accumulated_progress / track_obj.total_length) * 100

        print("\n" + "=" * 80)
        print("MCTS + PI-SVGP + CS + EXTRACT ACTIONS — TEST RESULT")
        print("=" * 80)
        print(f"Track:        {TRACK}")
        print(f"mu:           {MU}")
        print(f"Steps:        {step}")
        print(f"Reason:       {reason}")
        print(f"Collision:    {reason == 'collision'}")
        print(f"Laps:         {laps}")
        print(f"Progress:     {progress_pct:.1f}%")
        print(f"Avg step:     {avg_dt * 1000:.2f} ms")
        print(f"Lap times:    {lap_times_list}")
        print(f"CSV saved:    {csv_path}")
        print("=" * 80)

    finally:
        os.unlink(fixed_map_config_path)


if __name__ == "__main__":
    main()
