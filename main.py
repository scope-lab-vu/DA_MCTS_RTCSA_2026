"""
F1TENTH main loop.

Usage:
    python main.py --vehicle-params params/vehicle_params.yaml \
                   --algo-config   params/config_mppi_pinn.yaml \
                   --map-config    maps/config_example_map.yaml \
                   --render
"""

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.api import F1TenthAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1TENTH control loop")
    parser.add_argument(
        "--vehicle-params",
        type=Path,
        default=Path("params/vehicle_params.yaml"),
    )
    parser.add_argument(
        "--algo-config",
        type=Path,
        default=Path("params/config_simple_controller.yaml"),
    )
    parser.add_argument(
        "--map-config",
        type=Path,
        default=Path("maps/config_YasMarina_map.yaml"),
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=1.0489,
        help="Override friction coefficient (e.g. 0.4947, 0.8642, 1.0489)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable visualization of the gym environment",
    )
    parser.add_argument(
        "--localize",
        action="store_true",
        help="Run particle filter localization alongside the control loop and log errors",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility (matches reference MPPI script)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info("Random seed set to %d", seed)


def main() -> None:
    args = parse_args()

    # --- Set seed before anything else (read from algo config) ---
    from src.params import load_algo_config
    seed = load_algo_config(args.algo_config).simulation.seed
    set_seed(seed)

    # --- Build pipeline via API ---
    api = F1TenthAPI(args.vehicle_params, args.algo_config, args.map_config, mu_override=args.mu)
    waypoints = api.load_waypoints()
    render = args.render

    if render:
        api.setup_rendering(waypoints)

    dt_s = api.algo_config.simulation.dt_s
    dt_ms = dt_s * 1000.0
    log_every = api.algo_config.simulation.log_every_n_steps
    step = 0
    reason = "unknown"

    # Episode logger — records state-action pairs for CSV output.
    episode_logger = api.create_episode_logger()

    # --- Particle filter localization (optional) ---
    pf = None
    pf_logger = None
    prev_state = None
    scan_angles = None
    if args.localize:
        from src.localize.particle_filter import ParticleFilter, PFConfig
        from src.localize.particle_filter.pf_logger import PFLogger

        # Resolve map yaml path from map_config
        map_yaml = str(
            Path(api.map_config.map_path).parent
            / (Path(api.map_config.map_path).name + ".yaml")
        )
        log.info("Initializing particle filter with map: %s", map_yaml)

        pf_config = PFConfig(
            max_particles=1500,
            range_method='rm',  # CPU ray marching (no GPU needed in sim)
        )
        pf = ParticleFilter(pf_config, map_yaml)

        # Initialize at starting pose
        pf.set_pose(api.map_config.sx, api.map_config.sy, api.map_config.stheta)

        scan_angles = api.get_scan_angles()
        pf_logger = PFLogger(output_path="results/pf_localization_log.csv")
        log.info("Particle filter initialized with %d particles", pf_config.max_particles)

    # --- SVGP state error logging (when use_svgp_state is enabled) ---
    svgp_state_log = None
    use_svgp = api.algo_config.simulation.use_svgp_state
    if use_svgp:
        svgp_log_path = "results/svgp_state_error_log.csv"
        svgp_state_log = open(svgp_log_path, "w")
        svgp_state_log.write(
            "step,gt_x,gt_y,gt_delta,gt_v,gt_psi,gt_omega,gt_beta,"
            "svgp_x,svgp_y,svgp_delta,svgp_v,svgp_psi,svgp_omega,svgp_beta\n"
        )
        log.info("SVGP state logging enabled → %s", svgp_log_path)

    log.info("Starting control loop at dt=%.4fs (%.0f Hz)", dt_s, 1.0 / dt_s)

    try:
        while True:
            t_start = time.monotonic()

            # --- Core loop ---
            state = api.get_state()

            # --- Log SVGP state vs ground truth ---
            if svgp_state_log is not None:
                gt = api._vehicle._env.sim.agents[0].state
                svgp_state_log.write(
                    f"{step},{gt[0]:.8f},{gt[1]:.8f},{gt[2]:.8f},{gt[3]:.8f},"
                    f"{gt[4]:.8f},{gt[5]:.8f},{gt[6]:.8f},"
                    f"{state.x:.8f},{state.y:.8f},{state.delta:.8f},{state.v:.8f},"
                    f"{state.psi:.8f},{state.omega:.8f},{state.beta:.8f}\n"
                )

            # --- Particle filter update (if enabled) ---
            if pf is not None:
                scan = api.get_scan()

                # Compute odom delta in local frame
                if prev_state is not None:
                    dx_global = state.x - prev_state.x
                    dy_global = state.y - prev_state.y
                    dtheta = state.psi - prev_state.psi
                    # rotate into local frame of previous pose
                    cos_prev = np.cos(-prev_state.psi)
                    sin_prev = np.sin(-prev_state.psi)
                    dx_local = cos_prev * dx_global - sin_prev * dy_global
                    dy_local = sin_prev * dx_global + cos_prev * dy_global
                    odom_delta = np.array([dx_local, dy_local, dtheta])
                else:
                    odom_delta = np.array([0.0, 0.0, 0.0])

                pf_pose, pf_duration_ms = pf.update(scan, odom_delta, scan_angles)

                remaining_budget_ms = dt_ms - pf_duration_ms

                pf_logger.log(
                    gt_x=state.x, gt_y=state.y, gt_psi=state.psi,
                    pf_x=pf_pose[0], pf_y=pf_pose[1], pf_psi=pf_pose[2],
                    pf_duration_ms=pf_duration_ms,
                    dt_ms=dt_ms,
                )

            prev_state = state

            # Pass remaining budget (in seconds) to controller
            remaining_s = remaining_budget_ms / 1000.0 if pf is not None else 0.0
            cmd = api.compute_action(state, waypoints, remaining_budget_s=remaining_s)

            # Grab optional per-step diagnostics from controller
            ctrl = api._controller
            max_kappa = ctrl.get_max_kappa() if hasattr(ctrl, 'get_max_kappa') else float("nan")
            avg_kappa = ctrl.get_avg_kappa() if hasattr(ctrl, 'get_avg_kappa') else float("nan")
            c_alpha = ctrl.get_cornering_stiffness() if hasattr(ctrl, 'get_cornering_stiffness') else float("nan")

            episode_logger.log_step(state, cmd, max_kappa=max_kappa, avg_kappa=avg_kappa, c_alpha=c_alpha)
            done = api.apply_action(cmd)

            if render:
                api.render()

            step += 1

            # --- Periodic logging ---
            if step % log_every == 0:
                log.info(
                    "step=%d  x=%.3f y=%.3f psi=%.3f v=%.3f  accel=%.3f steer_vel=%.3f",
                    step,
                    state.x, state.y, state.psi, state.v,
                    cmd.acceleration, cmd.steering_vel,
                )

            if done:
                reason = "collision" if api.collision else "done"
                log.info("Episode finished at step %d (%s).", step, reason)
                break

            # --- Rate control with drift correction ---
            elapsed = time.monotonic() - t_start
            sleep_time = dt_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        reason = "interrupted"
        log.info("Shutting down after %d steps.", step)

    # --- Save trajectory ---
    api.save_episode(episode_logger, output_dir="results/", reason=reason)

    # --- Close PF logger ---
    if pf_logger is not None:
        pf_logger.close()
        log.info("PF localization log saved to results/pf_localization_log.csv")

    # --- Close SVGP state log and compute error ---
    if svgp_state_log is not None:
        svgp_state_log.close()
        # Compute and print RMSE
        import csv, math as _m
        with open("results/svgp_state_error_log.csv") as f:
            rows = list(csv.DictReader(f))
        if rows:
            n = len(rows)
            fields = ['x', 'y', 'delta', 'v', 'psi', 'omega', 'beta']
            print(f"\n=== SVGP State Error (RMSE over {n} steps) ===")
            for field in fields:
                gt_key, sv_key = f"gt_{field}", f"svgp_{field}"
                mse = sum((float(r[gt_key]) - float(r[sv_key])) ** 2 for r in rows) / n
                print(f"  {field:>8s}: {_m.sqrt(mse):.6f}")
        log.info("SVGP state error log saved to results/svgp_state_error_log.csv")


if __name__ == "__main__":
    main()
