"""
Typed parameter dataclasses and YAML loader.

Vehicle params and algorithm config are loaded from separate YAML files
into distinct dataclasses. This keeps physical constants decoupled from
algorithm-specific tuning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Vehicle parameters (from vehicle_params.yaml)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleParams:
    mu: float = 1.0489
    C_Sf: float = 4.718
    C_Sr: float = 5.4562
    lf: float = 0.15875
    lr: float = 0.17145
    h: float = 0.074
    m: float = 3.74
    I: float = 0.04712
    s_min: float = -0.4189
    s_max: float = 0.4189
    sv_min: float = -3.2
    sv_max: float = 3.2
    v_switch: float = 7.319
    a_max: float = 9.51
    v_min: float = -5.0
    v_max: float = 20.0
    width: float = 0.31
    length: float = 0.58

    def to_gym_dict(self) -> dict:
        """Convert to the dict format expected by f1tenth_gym."""
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# Algorithm config sub-sections (from config_*.yaml)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControllerConfig:
    type: str = "simple"
    target_speed_mps: float = 2.0
    # MPPI-specific fields
    horizon: int = 20
    num_samples: int = 500
    lambda_: float = 1.0
    noise_sigma_accel: float = 2.0
    noise_sigma_steer: float = 1.0
    v_ref: float = 5.0
    device: str = "cpu"
    use_occupancy_map: bool = True
    lane_width: float = 1.1
    real_track_width: float = 2.2
    # Cost function tuning
    v_max_cost: float = 11.15
    alpha_threshold: float = 3.5
    speed_weight: float = 5.0
    # MCTS DPW fields
    n_simulations: int = 500
    exploration_constant: float = 1.0
    alpha_dpw: float = 0.5
    beta_dpw: float = 0.5
    discount: float = 0.95
    rollout_horizon: int = 15
    rollout_horizon_straight: int = 5
    rollout_horizon_curve: int = 12
    kappa_threshold: float = 0.05
    a_lat_budget: float = 5.0
    curvature_lookahead: int = 30
    mcts_time_budget: float = 0.022
    # MCTS DPW PINN fields
    batch_size: int = 8
    eval_K: int = 4
    eval_device: str = "cuda"
    # MCTS DPW SVGP fields
    svgp_omega_model_path: Optional[str] = None
    svgp_beta_model_path: Optional[str] = None
    svgp_task_index: int = 0
    # MCTS DPW PI-SVGP fields
    pi_svgp_model_path: Optional[str] = None
    # PI-SVGP with C_alpha as direct input (no hidden params) model directory
    pi_svgp_no_hip_model_dir: Optional[str] = None
    # MCTS action prior SVGP model path
    action_prior_model_path: Optional[str] = None
    action_prior_model_type: str = "svgp"  # "svgp" or "sgpr_krasserm"
    # Normalize reward components to ~[0,1] scale
    normalize_reward: bool = False
    # Cornering stiffness adaptive velocity parameters (mppi_cs controller)
    # v_ref_t -= cs_w1 * log(cs_w2 * delta_C_alpha + cs_w3)
    # delta_C_alpha = max(0, C_alpha_th - cornering_stiffness)
    cs_w1: float = 1.0
    cs_w2: float = 0.25
    cs_w3: float = 0.0
    C_alpha_th: float = 104.89
    calpha_window_size: int = 10
    # Budget-aware MCTS adaptive action sampling
    budget_w_low: float = 15.0
    budget_w_high: float = 50.0
    budget_n_warmup: int = 100
    budget_n_recalibrate: int = 100


@dataclass(frozen=True)
class DynamicsConfig:
    type: str = "kinematic_bicycle"
    model_path: Optional[str] = None
    model_cfg_path: Optional[str] = None
    scaler_path: Optional[str] = None
    # SVGP-specific: separate omega/beta model paths and task index
    omega_model_path: Optional[str] = None
    beta_model_path: Optional[str] = None
    task_index: int = 0
    inference_device: str = "cpu"


@dataclass(frozen=True)
class SafetyConfig:
    type: str = "noop"


@dataclass(frozen=True)
class SimulationConfig:
    vehicle_interface: str = "gym"
    dt_s: float = 0.01
    log_every_n_steps: int = 100
    seed: int = 42


@dataclass(frozen=True)
class AlgoConfig:
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


# ---------------------------------------------------------------------------
# Map config (from maps/config_*_map.yaml)
# Paths in the yaml are relative to the config file's directory.
# The loader resolves them to absolute paths.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MapConfig:
    map_path: str = ""
    map_ext: str = ".png"
    sx: float = 0.0
    sy: float = 0.0
    stheta: float = 0.0
    wpt_path: str = ""
    wpt_delim: str = ";"
    wpt_rowskip: int = 3
    wpt_xind: int = 1
    wpt_yind: int = 2
    wpt_thind: Optional[int] = None
    wpt_vind: Optional[int] = None
    run_name: Optional[str] = None
    perf_num: Optional[int] = None
    centerline_path: Optional[str] = None
    track_name: Optional[str] = None
    mu_id: Optional[str] = None
    mu: Optional[str] = None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty YAML file: {path}")
    return data


def load_vehicle_params(path: Path) -> VehicleParams:
    data = _load_yaml(path)
    vehicle_section = data.get("vehicle", {})
    return VehicleParams(**vehicle_section)


def load_algo_config(path: Path) -> AlgoConfig:
    data = _load_yaml(path)
    return AlgoConfig(
        controller=ControllerConfig(**data.get("controller", {})),
        dynamics=DynamicsConfig(**data.get("dynamics", {})),
        safety=SafetyConfig(**data.get("safety", {})),
        simulation=SimulationConfig(**data.get("simulation", {})),
    )


def load_map_config(
    path: Path,
    mu_id_override: Optional[str] = None,
    mu_override: Optional[float] = None,
) -> MapConfig:
    """Load map config. Resolves relative paths against the config file's directory."""
    data = _load_yaml(path)
    config_dir = path.parent

    # Apply CLI overrides before template expansion.
    if mu_id_override is not None:
        data["mu_id"] = mu_id_override
    if mu_override is not None:
        data["mu"] = str(mu_override)

    # Expand {track_name} and {mu_id} placeholders in path fields.
    for key in ("map_path", "wpt_path", "centerline_path"):
        if key in data and data[key] is not None:
            data[key] = data[key].format_map(data)

    # Resolve relative paths to absolute.
    for key in ("map_path", "wpt_path", "centerline_path"):
        if key in data and data[key] is not None:
            data[key] = str((config_dir / data[key]).resolve())

    return MapConfig(**data)
