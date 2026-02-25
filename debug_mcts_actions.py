"""Quick debug: inspect MCTS root actions at the very start (v=0)."""
import sys, random
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, '/home/samir/Desktop/WORK/f1tenth_gym_ws')
from src.api import F1TenthAPI

VEHICLE_PARAMS = Path('params/vehicle_params.yaml')
BASE_CONFIG = Path('params/config_mcts_dpw_cpp_known_dyn.yaml')
MAP_CONFIG = Path('maps/config_YasMarina_map.yaml')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

api = F1TenthAPI(VEHICLE_PARAMS, BASE_CONFIG, MAP_CONFIG, mu_override=1.0)
waypoints = api.load_waypoints()

# Get access to the controller's tree
ctrl = api._controller

for step in range(30):
    state_msg = api.get_state()
    print(f"\n{'='*80}")
    print(f"Step {step}: v={state_msg.v:.4f}  psi={state_msg.psi:.4f}  "
          f"delta={state_msg.delta:.4f}")
    print(f"{'='*80}")

    cmd = api.compute_action(state_msg, waypoints)
    print(f"Chosen action: accel={cmd.acceleration:.4f}  sv={cmd.steering_vel:.4f}")

    # Inspect root actions from the tree
    tree = ctrl._tree
    if tree is not None:
        actions = tree.get_root_actions()
        # Sort by visits (most visited first)
        actions_sorted = sorted(actions, key=lambda a: a['visits'], reverse=True)
        print(f"\nRoot has {len(actions_sorted)} action branches:")
        print(f"  {'Accel':>8s}  {'SV':>8s}  {'Visits':>6s}  {'AvgReward':>12s}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*12}")
        for a in actions_sorted[:20]:  # top 20
            act = a['action']
            print(f"  {float(act[0]):+8.3f}  {float(act[1]):+8.3f}  "
                  f"{a['visits']:6d}  {a['avg_reward']:12.2f}")
    else:
        print("  (no tree — kickstart mode)")

    api.apply_action(cmd)
