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
    log_every = api.algo_config.simulation.log_every_n_steps
    step = 0
    reason = "unknown"

    # Episode logger — records state-action pairs for CSV output.
    episode_logger = api.create_episode_logger()

    log.info("Starting control loop at dt=%.4fs (%.0f Hz)", dt_s, 1.0 / dt_s)

    try:
        while True:
            t_start = time.monotonic()

            # --- Core loop ---
            state = api.get_state()
            cmd = api.compute_action(state, waypoints)
            episode_logger.log_step(state, cmd)
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


if __name__ == "__main__":
    main()
