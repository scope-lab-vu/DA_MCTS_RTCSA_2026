#!/usr/bin/env bash
#
# Launch 10 Optuna sweeps in parallel:
#   2 controllers (stanley_method, lqr) × 5 dt values (0.04, 0.06, 0.08, 0.1, 0.2)
# Each sweep:
#   - 3000 trials by default
#   - 3000 max_steps per trial
#   - logs to its own file under results/<ctrl>_optuna/
#   - own SQLite study DB (no shared-state contention)
#
# Each process runs detached (nohup) so you can close the shell and they
# survive. PIDs are saved to results/optuna_sweep_pids.txt for easy cleanup.
#
# Usage:
#   bash run_exps/launch_stanley_lqr_dt_sweep.sh
#   N_TRIALS=200 MAX_STEPS=1500 bash run_exps/launch_stanley_lqr_dt_sweep.sh
#   TRACK=Sepang bash run_exps/launch_stanley_lqr_dt_sweep.sh
#
# Monitor:
#   tail -f results/stanley_optuna/stanley_YasMarina_dt0.04_n3000.log
#   ps -p $(cat results/optuna_sweep_pids.txt | tr '\n' ' ') -o pid,etime,command
# Kill all:
#   kill $(cat results/optuna_sweep_pids.txt)
#
set -euo pipefail

# Repo root (script lives in run_exps/).
cd "$(dirname "$0")/.."

# Pin to the venv's python by absolute path so this works regardless of
# how the script was invoked (sh / bash / cron / from a fresh shell).
PYTHON="$(pwd)/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "FATAL: $PYTHON not found / not executable. Did you create the venv?" >&2
  exit 1
fi

DT_VALUES=(0.04 0.06 0.08 0.1 0.2)
N_TRIALS=${N_TRIALS:-3000}
MAX_STEPS=${MAX_STEPS:-3000}
TRACK=${TRACK:-YasMarina}
MU=${MU:-1.0489}

mkdir -p results/stanley_optuna results/lqr_optuna

PIDS_FILE="results/optuna_sweep_pids.txt"
> "$PIDS_FILE"  # clear previous

echo "============================================================"
echo "Launching 10 sweeps: 2 ctrls × 5 dt values"
echo "  TRACK=$TRACK  MU=$MU  N_TRIALS=$N_TRIALS  MAX_STEPS=$MAX_STEPS"
echo "============================================================"

launch_one() {
  local script="$1"        # run_exps/run_stanley_optuna.py | run_exps/run_lqr_optuna.py
  local ctrl="$2"          # stanley | lqr
  local dt="$3"
  local log="results/${ctrl}_optuna/${ctrl}_${TRACK}_dt${dt}_n${N_TRIALS}.log"
  echo "  → $ctrl  dt=$dt  log=$log"
  nohup "$PYTHON" "$script" \
    --n "$N_TRIALS" \
    --max-steps "$MAX_STEPS" \
    --track "$TRACK" \
    --mu "$MU" \
    --dt-s "$dt" \
    > "$log" 2>&1 &
  local pid=$!
  echo "$pid  $ctrl  dt=$dt  $log" >> "$PIDS_FILE"
}

for dt in "${DT_VALUES[@]}"; do
  launch_one run_exps/run_stanley_optuna.py stanley "$dt"
done
for dt in "${DT_VALUES[@]}"; do
  launch_one run_exps/run_lqr_optuna.py     lqr     "$dt"
done

echo
echo "Launched $(wc -l < "$PIDS_FILE" | tr -d ' ') sweeps."
echo "PIDs + logs:  $PIDS_FILE"
echo
echo "Monitor a single sweep:"
echo "  tail -f results/stanley_optuna/stanley_${TRACK}_dt0.04_n${N_TRIALS}.log"
echo
echo "Status of all:"
echo "  ps -p \$(awk '{print \$1}' $PIDS_FILE | tr '\n' ',' | sed 's/,\$//') -o pid,etime,%cpu,command"
echo
echo "Kill all:"
echo "  kill \$(awk '{print \$1}' $PIDS_FILE)"
echo
echo "(Script exits now; sweeps run in background.)"
