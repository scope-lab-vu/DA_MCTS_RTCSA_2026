# MCTS DPW C++ — Fast Tree Search for F1TENTH Racing

C++ accelerated Monte Carlo Tree Search with Double Progressive Widening for the F1TENTH autonomous racing platform. Functionally identical to the Python `mcts_dpw_py` controller but **~200x faster** thanks to compiled C++ dynamics, rollouts, and tree search.

## Performance

| Metric | Python (`mcts_dpw_py`) | C++ (`mcts_dpw_cpp`) | Speedup |
|--------|------------------------|----------------------|---------|
| Avg step time (500 sims) | ~865 ms | ~4 ms | **~210x** |

## Build

Requires a C++17 compiler and pybind11.

```bash
# From the workspace root (f1tenth_gym_ws/)
pip install -e src/controller/mcts_dpw_cpp/
```

Verify the build:
```bash
python -c "from mcts_f1tenth_cpp import F1TenthDPW; print('OK')"
```

## Usage

Use the config file `params/config_mcts_dpw_cpp_known_dyn.yaml`:

```bash
python main.py \
    --vehicle-params params/vehicle_params.yaml \
    --algo-config params/config_mcts_dpw_cpp_known_dyn.yaml \
    --map-config maps/config_YasMarina_map.yaml \
    --mu 1.0489 \
    --render
```

Or change `type: mcts_dpw_py` to `type: mcts_dpw_cpp` in any existing MCTS config — all hyperparameters are identical.

## Architecture

```
Python (controller.py)                 C++ (mcts_f1tenth_cpp.so)
┌──────────────────────┐              ┌───────────────────────────┐
│ MCTSDPWCppController │              │ F1TenthDPW                │
│                      │   pybind11   │  ├── DPWNative (tree)     │
│  Track loading       │ ──────────►  │  │   ├── SPWNative        │
│  Occupancy map build │              │  │   └── MCTSNative       │
│  Ref trajectory      │              │  └── F1TenthSimulator     │
│  v_ref window calc   │              │       ├── Dynamics (C++)  │
│                      │              │       ├── Cost function   │
│  learn() / forward() │ ◄──────────  │       └── Occupancy map  │
│  best_action()       │              │           (O(1) lookup)   │
└──────────────────────┘              └───────────────────────────┘
```

**Key design decisions:**

- **Dynamics fully in C++**: The F1TENTH single-track dynamics (kinematic + dynamic model) are ported to C++ header-only code. No Python calls during tree search.
- **Occupancy map passed from Python**: Built once using scipy/numpy in Python, then the flat grid array is passed to C++ at construction. O(1) off-track lookup via integer indexing.
- **Reference velocity window**: Computed per step in Python (requires track data), passed to C++ before each `learn()` call.
- **Tree reuse**: Identical to the Python version — `forward()` advances the root node after each real step.

## Files

```
src/controller/mcts_dpw_cpp/
├── __init__.py              # Package export
├── controller.py            # Python wrapper (same interface as mcts_dpw_py)
├── setup.py                 # pybind11 build script
├── pyproject.toml           # Build system metadata
├── include/
│   ├── f1tenth_simulator.hpp  # F1TENTH dynamics + reward (C++)
│   ├── mcts_native.hpp        # DPW/SPW/MCTS tree search (C++)
│   ├── nodes.hpp              # Decision/Random node structs
│   └── simulator.hpp          # Abstract simulator interface
└── src/
    └── bindings.cpp           # pybind11 Python bindings
```

## Hyperparameters

All hyperparameters are identical to the Python version and configured in the YAML config:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_simulations` | 500 | Tree traversals per action selection |
| `exploration_constant` | 1.0 | UCB exploration constant (K) |
| `alpha_dpw` | 0.5 | Action progressive widening exponent |
| `beta_dpw` | 0.5 | Outcome progressive widening exponent |
| `discount` | 0.99 | Rollout discount factor |
| `rollout_horizon` | 15 | Max rollout steps |
| `speed_weight` | 100.0 | Speed cost multiplier |
| `alpha_threshold` | 1.0 | Slip angle penalty threshold (rad) |
| `v_ref` | 10.0 | Fallback reference velocity (m/s) |
| `lane_width` | 1.1 | Track width for occupancy map (m) |
