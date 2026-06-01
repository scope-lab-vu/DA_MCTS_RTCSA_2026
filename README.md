# DA-MCTS: Deadline-Aware Action Exploration for Safe Real-Time Planning

Source code for the RTCSA 2026 paper *"DA-MCTS: Deadline-Aware Action Exploration
for Safe Real-Time Planning"*. DA-MCTS is a continuous-action Monte Carlo Tree
Search planner for F1TENTH autonomous racing that adapts its action sampling to
the available compute budget (relative deadline), using an SVGP dynamics model,
a learned Gaussian action prior, and an MLP that predicts sampling weights online.

All model weights, maps, and trajectories needed to run the experiments are
already included in this repository — no extra downloads are required.

---

## 1. Requirements

- **Python 3.8+**
- A **C++17 compiler**
- Tested on macOS (Apple Silicon) and Linux x86-64.

On Ubuntu, install the OS-level build prerequisites first (needed to compile the
C++ planner, otherwise step 5 fails with `Python.h: No such file or directory`):

```bash
sudo apt update
sudo apt install -y build-essential python3.8-dev python3.8-venv
```

## 2. Install

Run these commands from the repository root.

```bash
# 1. Create and activate a Python 3.8 virtual environment
python3.8 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Build gym 0.19.0 
pip install gym==0.19.0

# 3. Install the remaining Python dependencies
pip install -r requirements.txt

# 4. Install the F1TENTH simulator (editable)
pip install -e f1tenth_gym

# 5. Build and install the DA-MCTS C++ planner extension (editable)
pip install -e src/controller/da_mcts/
```

### Verify the install
 
```bash
python -c "import da_mcts, f110_gym, torch, gpytorch; print('install OK')"
```

## 3. Quick sanity run (~3–4 min)

Run a single, easy configuration (high friction, one track, one trial) and check
that the car completes laps without crashing:

```bash
python run_exps/run_da_mcts_experiments.py \
    --track YasMarina --mu 1.0 --time-budget 0.030 \
    --num-trials 1 --max-steps 4000
```

A summary table is printed at the end. A successful run shows `Collision = No`
and `Laps >= 2` with per-lap times.

## 4. Reproduce the paper experiments

Results are written to `results/` (one subfolder per track / friction / budget).

```bash
# Main DA-MCTS sweep: 2 tracks x 10 friction values x 7 time budgets x 10 trials
python run_exps/run_da_mcts_experiments.py

# Uniform-sampling ablation
python run_exps/run_da_mcts_experiments_ablation_uniform.py
```

Both scripts accept the same filters, useful for running a subset:

| Flag | Meaning | Example |
|------|---------|---------|
| `--track`        | one or more tracks (`Sepang`, `YasMarina`) | `--track YasMarina` |
| `--mu`           | one or more friction values                | `--mu 1.0 0.5 0.4` |
| `--time-budget`  | one or more MCTS budgets, in **seconds**   | `--time-budget 0.030` |
| `--num-trials`   | trials (seeds) per configuration           | `--num-trials 10` |
| `--max-steps`    | max simulation steps per episode           | `--max-steps 10000` |
| `--render`       | show the live simulation window            | `--render` |

Run with `--help` to see all options.
