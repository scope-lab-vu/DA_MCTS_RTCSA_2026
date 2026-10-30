# Results and limitations

Clean run: YasMarina, mu = 1.0, **5 seeds per cell** (42-46), planner pinned to one CPU core,
time budget T = 25 ms, control period 40 ms, fixed-work arm N = 900 iterations, performance
governor, AC power, turbo boost off. All numbers below are copied from
`docs/results/condition_table.csv`, `docs/results/timing_only/condition_table.csv` and
`docs/results/post_hoc_checks.txt` (produced by `analysis/summarize_results.py` and
`analysis/post_hoc_checks.py`). Commit hash of the code used: **TODO**.

## Headline

With the released DA-MCTS, a time-bounded search did **not** make the control step meet its
deadline under CPU contention, and deadline-aware action sampling gave no measurable deadline
or safety benefit over simply stopping the search on time. In closed loop, every planner
(fixed-work, anytime, DA-MCTS) crashed in all contended episodes once late plans were
modelled. The main reason is that the time budget bounds only the search, not the rest of the
control step.

## 1. The contention behaves as designed

| Measure | Idle | Light (1 competing worker) | Heavy (3 competing workers) |
|---|---|---|---|
| Fixed-work planner, median plan time | 31.1 ms | 64.6 ms (2.08x) | 134.0 ms (4.31x) |
| Anytime, iterations per 25 ms budget | 957 | 493 (0.52x) | 276 (0.29x) |
| DA-MCTS, iterations per 25 ms budget | 966 | 507 (0.53x) | 294 (0.30x) |

(Timing-only variant, late model off.) Iteration counts fall in proportion to the CPU share the
planner receives.

## 2. Deadline reliability (timing-only variant; deadline = 40 ms per control step)

Hit rate by decision / by wall-clock time (a slower planner makes fewer decisions while
overloaded, so both views are shown):

| Arm | Idle | Light | Heavy | Bursty (decision) | Bursty (time) |
|---|---|---|---|---|---|
| Fixed work | 1.000 | 0.000 | 0.000 | 0.796 | 0.496 |
| Anytime | 1.000 | 0.071 | 0.000 | 0.667 | 0.503 |
| DA-MCTS | 1.000 | 0.076 | 0.000 | 0.642 | 0.478 |

- Under sustained load the fixed-work planner meets essentially no deadlines, and the
  time-bounded planners meet only about 7% at light load and none at heavy load.
- The apparent bursty difference by decision count (0.80 vs 0.64) disappears when measured by
  time (0.50 vs 0.48): all arms are on time roughly when the load is off. It was a counting
  artifact.
- DA-MCTS minus fixed-work at light load: +0.076 (95% CI 0.060 to 0.097). Statistically
  detectable, practically small.

**Why a time budget does not fix it.** The configured budget (25 ms) covers only the C++ search.
The median whole-step time was 32.6 ms idle, 43.3 ms at light and 66.6 ms at heavy load, so the
work outside the timer is about 7.5 ms idle, 18 ms at light and 42 ms at heavy (median step
time minus budget). At heavy load that work alone exceeds the 40 ms period, so no search
budget could meet it on this hardware.

## 3. Safety under late plans (closed-loop variant; a late plan means the car keeps its previous command)

| Arm | Idle: completed / collided | Light | Heavy | Bursty |
|---|---|---|---|---|
| Fixed work | 5/5, 0 | 0/5, 5 collisions | 0/5, 5 | 0/5, 5 |
| Anytime | 4/5, 1 | 0/5, 5 | 0/5, 5 | 0/5, 5 |
| DA-MCTS | 4/5, 1 | 0/5, 5 | 0/5, 5 | 0/5, 5 |

Every contended episode ended in a collision after roughly 55-180 steps. This floor effect means
the closed-loop variant cannot rank the arms under contention; it shows that late commands are
very costly at this speed.

## 4. Deadline-aware sampling vs. uniform anytime sampling (timing-only variant)

- Idle lap time: DA-MCTS 48.8 s, anytime 75.4 s, fixed work 79.7 s. The learned weights make
  the car drive faster even without contention (a sampling-policy effect, not a contention effect).
- With reduced iterations, DA-MCTS completed fewer episodes than anytime: light 3/5 vs 5/5,
  bursty 3/5 vs 5/5, heavy 2/5 vs 3/5. Intervals are wide at n = 5 (for example, progress
  difference at light -44 percentage points, CI -96 to 0), so this is a possible speed-versus-safety
  trade-off, not an established effect.
- No arm showed a reliable deadline advantage over another beyond the small light-load difference.

## 5. Hypotheses

| Hypothesis | Outcome |
|---|---|
| H1: DA-MCTS meets more deadlines than fixed work under contention | Not supported in practice (+0.076 at light, 0 at heavy, no difference by time at bursty) |
| H2: DA-MCTS keeps better progress and completion under contention | Not assessable: all arms fail in every contended closed-loop episode |
| H3: the gap widens as contention increases | Not testable: the fixed-work baseline already fails at light load |

## 6. Limitations

- **Single machine and track.** One 8 GB Fedora laptop, one track (YasMarina), one friction value,
  simulation only.
- **Few seeds.** Five per cell, so confidence intervals are wide; differences smaller than the
  intervals should not be interpreted.
- **Budget tuned at idle only.** T = 25 ms was chosen from an idle calibration. A budget chosen
  for each contention level, or one that subtracts the measured non-search time, was not
  tested. The results describe a 25 ms budget on this hardware.
- **Slow regime.** Turbo boost was disabled for stable timings. Together with the other
  machine-state changes, iterations per 25 ms fell from about 2600 (first calibration, turbo on)
  to about 960, roughly 2.7x slower. Conclusions may differ on faster hardware, where the
  non-search overhead is smaller.
- **First sweep discarded.** A first 10-seed sweep ran before the machine state was controlled
  (battery power, powersave governor, turbo on, browser open). Its idle baseline drifted, and it
  is not reported here. It also gave different closed-loop outcomes at light load, which shows how
  sensitive the conclusions are to machine speed.
- **Memory.** The machine had 7.4 GiB of RAM, with 1.4-1.7 GB held in compressed swap
  throughout. System-wide memory stall, logged per episode, had a median of 2% of episode time at
  idle and 3-5% under load (maximum 11%; 40 of 119 episodes above 5%), over a window that includes
  the one-off map construction. The one large idle-cell timing spike (941 ms, 1 of 32 idle
  episodes) occurred in an episode with almost no memory stall (0.3%), so memory pressure was
  not identified as its cause. The contention results are consistent with CPU share alone, so
  they are not attributed to memory pressure, but a higher-memory machine would be needed to
  exclude it.
- **Power state flag.** The logged AC flag read 0 in 81 of 238 snapshots with the logging code
  used for this run; this may be a logging error (it kept only one of several power-supply
  entries) or real battery periods. Idle fixed-work plan times were constant at 31.0-31.1 ms across
  all idle cells, so no timing effect was visible.
- **Late-command model.** Holding the previous command is a simplification of real actuation
  delay, and the simulator itself does not wait for or react to the planner's wall-clock time.
- **Non-determinism.** The planner depends on wall-clock iterations, so identical seeds do not
  reproduce identical trajectories.
- **Hit-rate definition.** Deadline = whole planning call within the 40 ms control period, not
  the nominal search budget.

## 7. Future work

1. An overhead-aware budget: set the search budget each step to the deadline minus the measured
   non-search time.
2. Budget sensitivity (for example 10, 15, 20 ms) under light and bursty load.
3. A lighter or faster non-search stage (curvature, action-prior inference), since it dominates
   under heavy load.
4. More seeds, a second track and friction value, and a higher-memory or dedicated machine.
5. A finer load scale (several intermediate contention levels) so that H3 becomes testable.
