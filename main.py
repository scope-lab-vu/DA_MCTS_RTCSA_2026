"""
F1TENTH main loop.

Usage:
    python main.py --vehicle-params params/vehicle_params.yaml \
                   --algo-config   params/config_mppi_pinn.yaml
"""

import argparse
import logging
import time
from pathlib import Path

from src.api import F1TenthAPI
from src.idl_messages import WaypointsMsg

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
        default=Path("params/config_mppi_pinn.yaml"),
    )
    return parser.parse_args()


def make_demo_waypoints() -> WaypointsMsg:
    """Simple straight-line waypoints for testing the pipeline."""
    return WaypointsMsg(
        x=[float(i) for i in range(0, 20)],
        y=[0.0] * 20,
    )


def main() -> None:
    args = parse_args()

    # --- Build pipeline via API ---
    api = F1TenthAPI(args.vehicle_params, args.algo_config)
    waypoints = make_demo_waypoints()

    dt_s = api.algo_config.simulation.dt_s
    log_every = api.algo_config.simulation.log_every_n_steps
    step = 0

    log.info("Starting control loop at dt=%.4fs (%.0f Hz)", dt_s, 1.0 / dt_s)

    try:
        while True:
            t_start = time.monotonic()

            # --- Core loop ---
            state = api.get_state()
            cmd = api.compute_action(state, waypoints)
            api.apply_action(cmd)

            step += 1

            # --- Periodic logging ---
            if step % log_every == 0:
                log.info(
                    "step=%d  x=%.3f y=%.3f psi=%.3f v=%.3f  accel=%.3f steer_vel=%.3f",
                    step,
                    state.x, state.y, state.psi, state.v,
                    cmd.acceleration, cmd.steering_vel,
                )

            # --- Rate control with drift correction ---
            elapsed = time.monotonic() - t_start
            sleep_time = dt_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log.info("Shutting down after %d steps.", step)


if __name__ == "__main__":
    main()
