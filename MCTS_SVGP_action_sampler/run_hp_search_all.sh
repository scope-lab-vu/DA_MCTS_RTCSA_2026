#!/usr/bin/env bash
# Run HP search for all three feature sets sequentially.
# Usage: bash MCTS_SVGP_action_sampler/run_hp_search_all.sh

set -euo pipefail

PYTHON="/media/vanderbilt/home/sgupta/WORK/f1tenth_gym_ws/venv/bin/python"
SCRIPT="MCTS_SVGP_action_sampler/hp_search_krasserm.py"
DATA_DIR="MCTS_SVGP_action_sampler/training_data"
N_TRIALS=100

for fs in no_avg_kappa no_lateral general; do
    echo "========================================"
    echo "  HP search: feature_set=${fs}"
    echo "========================================"
    ${PYTHON} ${SCRIPT} \
        --data-dir ${DATA_DIR} \
        --feature-set ${fs} \
        --n-trials ${N_TRIALS} \
        2>&1 | tee "hp_search_${fs}.log"
    echo ""
done

echo "All three HP searches complete."
echo "Results saved in:"
echo "  models/sgpr_krasserm_no_avg_kappa/"
echo "  models/sgpr_krasserm_no_lateral/"
echo "  models/sgpr_krasserm_general/"
echo "Logs saved in:"
echo "  hp_search_no_avg_kappa.log"
echo "  hp_search_no_lateral.log"
echo "  hp_search_general.log"
