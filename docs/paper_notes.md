# Notes on the DA-MCTS paper

Gupta, Yuhas, Dubey (Vanderbilt University). "DA-MCTS: Deadline-Aware Action Exploration
for Safe Real-Time Planning." RTCSA 2026. Notes written from my reading of the paper;
page-level citations still to be added.

## Problem the paper addresses

Planners tuned on powerful hardware can become unsafe on resource-constrained hardware:
with fewer iterations, aggressive actions are sampled but not explored enough, so their
value estimates are unreliable and the car can crash. A permanent conservative fallback
(Simplex) is safe but gives up the benefit of planning.

## Method in brief

- Continuous-action MCTS with double progressive widening (DPW) on F1TENTH.
- **Three action-sampling categories:** aggressive (acceleration uniform in [0, a_max]),
  cautious (uniform in [a_min, 0]), and Gaussian-prior (acceleration from a learned
  Gaussian process). Steering velocity always comes from the Gaussian prior.
- **A small MLP** maps runtime features z = (speed, max curvature ahead, iterations done,
  max tree depth, time remaining) to the three sampling probabilities. The paper
  describes two hidden layers of 32 units with softmax output, and says the probabilities
  are recalibrated every N_r iterations (N_r = 10 in the paper's hyperparameter table).
- The MLP was trained on data from planning budgets of 1 to 40 ms, with labels from an
  oracle that depends on an estimated tree width (W_low = 15, W_high = 50).
- **Dynamics:** extended kinematic model plus a sparse variational Gaussian process (SVGP)
  that learns the residual in yaw rate and slip angle, with an estimated cornering
  stiffness as an extra input to adapt to friction. The paper reports that a PINN-based
  dynamics model took about 26 s per planning step against a 40 ms deadline.

## Evaluation in the paper

- Tracks YasMarina and Sepang; friction 0.1 to 1.0; 10 episodes of 2 laps per setting.
- Baselines (fixed 40 ms budget): MPC, PINN-based planner, MAP (pursuit controller), and
  a Simplex-style fallback limited to 1.75 m/s.
- DA-MCTS was also run at fixed budgets from 15 to 40 ms and with a dynamically changing
  budget caused by a particle-filter localiser running before it (10-15 ms per step).
- Reported on the Jetson Orin Nano: DA-MCTS adapts its time between 1 and 40 ms; MPC about
  81 ms, PINN about 500 ms per action.
- Reported ablations: uniform sampling gave comparable crash counts but slower laps;
  removing the cornering-stiffness input gave 54 crashes in 200 trials.
- Reported limitation: at mu = 0.1 on YasMarina the car crashes consistently.

## What I want to add (to verify against the paper before claiming it)

To my reading, the paper does not report a per-step **deadline hit rate**, and does not
compare DA-MCTS against a **fixed-work planner without a clock** under a **controlled,
repeatable contention schedule**. Its baselines are other controllers, and its one
dynamic-budget experiment uses a particle filter. This study adds exactly that comparison.

## Things I observed in the released code that differ from or go beyond the paper

(From my code reading with an AI-generated guide; each needs checking.)

- The time budget covers only the C++ search. Python preparation and action extraction
  run outside it, adding roughly 8-10 ms on my machine (measured).
- The simulation loop does not advance while the planner runs, so a late plan has no
  effect on the car unless a delay model is added.
- The first real planning call took about 3.9 s in my runs (setup cost).
- `run_da_mcts_experiments_ablation_uniform.py` may not enable uniform sampling
  (unverified).
- Run-to-run outcomes differ for the same seed because the planner depends on
  wall-clock iterations: in three runs on the same idle machine with seed 42 I saw a
  completed 2-lap run, a crash at 51% progress, and another completed run with a very
  different second lap time.

## Related work from the paper's reference list (not yet read by me)

- Baier and Winands, "Time management for Monte Carlo tree search" (2016).
- Sagers, Winands, Soemers, "Anytime sequential halving in Monte-Carlo tree search" (2025).
- Choudhury, Gupta, Morales, Kochenderfer, "Scalable anytime planning for multi-agent
  MDPs" (2021).
- Seto, Krogh, Sha, Chutinan, "The Simplex architecture for safe online control system
  upgrades" (1998).
- McMahan and Zhu, "Anytime-constrained reinforcement learning" (2024).
- Wang, Xia, Chung, Greenwald, "Dynamic Thinker: optimizing decision-time planning with
  costly compute" (2025).
- Soyyigit, Yao, Yun, "MURAL: multi-resolution anytime framework for LiDAR object
  detection" (RTCSA 2025).

## Reading plan

1. Baier and Winands (closest to "allocate time across decisions").
2. Seto et al. (what the baseline of safe fallback looks like).
3. One real-time scheduling reference on deadline miss ratio as a metric
   (to be found; my own suggestion, not from the paper).
