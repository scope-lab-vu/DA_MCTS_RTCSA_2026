# Methodology and research protocol

## 1. Research question

Can a deadline-aware anytime planner preserve higher trajectory quality and meet
real-time planning deadlines more reliably than a fixed-computation-budget planner
when CPU availability is unpredictable?

Formally: under controlled but unpredictable CPU contention, does adapting an anytime
planner's computation to the remaining deadline improve the trade-off between on-time
trajectory quality and deadline reliability, compared with a fixed-work baseline?

## 2. Why this gap exists

Planners are usually evaluated assuming either a fixed computation budget per control
step, or idealised compute availability. On real systems, CPU contention, thermal
throttling, OS scheduling and background load make planning time vary. A planner that
drives well in simulation but misses its deadlines under such variation is not useful in
practice. The DA-MCTS paper adapts *what the planner explores* to the time left, but, to
my reading, evaluates it across fixed budgets and one particle-filter test rather than
under a controlled contention schedule compared against a fixed-work baseline. This
study fills that specific gap. (Verify this claim against the paper before publishing.)

## 3. Planner arms

| Arm | Meaning | Configuration |
|---|---|---|
| `fixed` | **Fixed-work baseline.** Exactly N MCTS iterations per step, no clock check. Under contention the same work takes longer. | `mcts_time_budget=0`, `n_simulations=N`; sampling weights fixed at 1/3 each |
| `anytime` | **Ablation.** Time-bounded search (stops at the budget) with uniform sampling weights. Isolates the effect of stopping on time. | `mcts_time_budget=T`, `budget_n_warmup` set very large so the MLP never adapts the weights (verify in the C++ code) |
| `da` | **Method under study.** Time-bounded search with deadline-aware sampling weights from the MLP. | repository default with `mcts_time_budget=T` |

Comparing `fixed` with `da` answers the research question. Adding `anytime` separates
two effects that would otherwise be confounded: stopping on time, and adapting the
action sampling. The repository script `run_da_mcts_experiments_ablation_uniform.py`
is **not** used as a baseline: it is still time-bounded, and my code reading found
that it does not appear to enable uniform sampling (unverified; see CODEBASE_GUIDE.md,
section M).

**Choosing N.** N is the 10th percentile of the iteration counts that `da` achieves on
an idle machine (after warm-up), rounded down to 50, so the fixed arm just meets the
deadline when the machine is idle. This is produced by the `calibrate` command.

## 4. Definitions

- **Control period / deadline D.** 40 ms (`dt_s` in the simulator configuration).
- **Planning time.** Wall-clock time of the whole `compute_action` call, including the
  Python preparation (curvature, Gaussian-process prior, cornering stiffness), tree
  handling, the C++ search and action extraction. The configured time budget T only
  covers the C++ search, so the whole call is longer than T.
- **Deadline hit.** A planning step with planning time <= D. **Deadline hit rate** is
  the fraction of steps that hit, after warm-up.
- **Severe miss.** Planning time > 2 D. **Worst-case runtime** is the maximum planning
  time over an episode.
- **Late-command model.** The simulator does not advance while the planner runs, so
  lateness would otherwise be invisible. If a plan takes longer than D, the car keeps
  applying its previous command for the extra periods:
  `late_steps = ceil(plan_time / D) - 1`. The new command, computed from the older
  state, is applied afterwards.
- **Warm-up.** The first 40 decisions are excluded from deadline statistics and from the
  late model. The first real planning call takes seconds (tree build and cache warm-up);
  this is setup cost, not a missed deadline.
- **Collision rate.** Fraction of episodes ending in a collision.
- **Lap completion rate.** Fraction of episodes that finish all laps without a crash.
- **Progress.** Percent of the track completed (can exceed 100% for two laps).
- **Steps before crash.** Simulated steps until collision, for crashed episodes.
- **Smoothness.** Mean absolute change of the applied steering velocity and acceleration
  between consecutive simulated steps (lower is smoother). Held commands count.
- **Episode cap.** `max_steps = 6000` simulated steps (240 s). Uniform-weight planners
  drive more slowly (about 85-90 s per lap at idle in the pilot) and need more than the
  original 4000 steps for two laps, which would score slow-but-safe runs as incomplete.
- **Change relative to own idle.** The arms use different sampling policies and so drive
  at different speeds even without contention. The effect of contention is therefore
  reported as each arm's change from its own idle cell (`relative_to_idle.csv`), in
  addition to the cross-arm comparisons.
- **Lap time.** Mean and best lap time of completed episodes only. Read it together with
  completion rate, because crashed episodes have no lap time (survivorship bias).

## 5. Experimental variables

Primary study (this repository's default configuration):

| Variable | Values |
|---|---|
| Track | YasMarina |
| Friction mu | 1.0 |
| Time budget T | one value chosen after calibration (default 25 ms) |
| Planner arm | `fixed`, `anytime`, `da` |
| CPU condition | `none`, `light`, `heavy`, `bursty` |
| Seeds | 10 per condition, identical across arms |

That is 3 arms x 4 conditions x 10 seeds = 120 episodes, roughly 8 hours.

Optional extensions, run only after the primary study is analysed: a second track
(Sepang), lower friction (for example 0.7), and a budget sweep (15, 20, 25, 30 ms) to
produce budget-sensitivity curves.

### 5.1 Two variants of every cell

Every (arm, load) cell is run in two variants, because one cannot answer both questions:

| Variant | Late-command model | Tag in results | What it answers |
|---|---|---|---|
| closed-loop | on | `<load>` | Does the car crash or complete when plans arrive late? |
| timing-only | off | `<load>-timing` | How reliably does each planner meet the deadline? (the car never drives on stale commands, so episodes run long and give thousands of decisions) |

Reason: under heavy load the fixed-work arm can crash within a few decisions in the
closed-loop variant (observed in the pilot: a crash at step 58 after only 9 post-warm-up
decisions). Its deadline statistics then rest on a handful of samples and cannot be
compared. The timing-only variant provides the deadline and runtime statistics (H1);
the closed-loop variant provides safety and completion (H2). Report each metric from the
variant designed for it.

## 6. Controlled CPU contention

The planner process is pinned to **one core** with `taskset`. Contention is created by
busy-loop workers pinned to the **same** core (`experiments/cpu_load.py`). Pinning to one
core matters: if the planner may use several cores and only one is loaded, the operating
system moves the planner to a free core and the load has no effect.

| Condition | Competing workers on the planner's core | Expected CPU share for the planner |
|---|---|---|
| `none` | 0 | about 100% |
| `light` | 1, steady | about 50% |
| `heavy` | 3, steady | about 25% |
| `bursty` | 3, random on/off, 0.5-3.0 s each, seeded | varies between 25% and 100% |

The `bursty` schedule is unpredictable to the planner but reproducible: the same seed
gives the same schedule for every arm. Each worker's start and end times are written to
`results/load_logs/`. The CPU shares above are expectations from Linux fair scheduling,
not measurements.

Machine preparation: CPU governor set to `performance`, laptop plugged in, no other
heavy programs, trials run one at a time (never in parallel), numeric-library threads
limited to one.

## 7. Protocol

1. `python experiments/cpu_contention_experiments.py calibrate` (idle, pinned core).
   Check the idle deadline hit rate. If it is below about 97%, lower `time_budget_s`
   and calibrate again. Record the final values.
2. Pilot: `run --seeds 3`. Check that crashes, hit rates and lap times look sensible and
   that the contention conditions actually change the planning-time distribution
   (`runtime_distribution.png`). If `light` and `heavy` look identical to `none`, the
   contention is not working; fix that before continuing.
3. Full run: `run`. The sweep is resumable. Arm order is randomised per seed and
   condition (reproducibly) to avoid systematic drift effects.
4. Analyse: `analysis/summarize_results.py`, then `analysis/generate_plots.py`.

## 8. Statistics

- Per condition: mean, standard deviation, median, and 95% confidence intervals
  (Wilson for proportions such as collision and completion rate, bootstrap for means).
- Comparisons are **paired by seed** within each load condition. For each metric the
  difference (arm A minus arm B) is computed per seed; the bootstrap 95% interval of its
  mean is reported. A difference is called reliable only if the interval excludes zero.
- Comparisons: `da` vs `fixed` (the research question), `da` vs `anytime` (sampling
  effect only), `anytime` vs `fixed` (effect of stopping on time).
- With 10 seeds, intervals are wide. Report them; do not report point estimates alone.

## 9. Hypotheses

- **H1.** Under equal nominal budgets, `da` has a higher deadline hit rate than `fixed`
  when contention increases.
- **H2.** `da` keeps better progress and lap completion than `fixed` under contention.
- **H3.** The `da` minus `fixed` gap widens as the compute environment becomes less
  predictable (none, then light, then heavy; `bursty` is reported separately).

A time-bounded planner meets deadlines largely by construction, so H1 may be partly
trivial for `anytime` and `da` versus `fixed`. The informative comparisons are `da` vs
`anytime` and H2.

## 10. Threats to validity (state these in any write-up)

- Simulation only; contention generated on one machine, not on embedded hardware.
- The late-command model simplifies real actuation delay.
- The planner's behaviour depends on wall-clock iterations, so identical seeds do not
  give identical trajectories; variability across runs is real and is why seeds are
  replicated.
- The time budget, N, and the warm-up length are tuned on one idle machine and might not
  transfer.
- Lap times exclude crashed episodes (survivorship bias).
- One track and friction value in the primary study; conclusions do not extend beyond
  them without the extensions.
- Work outside the C++ timer is not controlled by the time budget, so planning time can
  exceed the deadline even for a time-bounded planner.

## 11. Reproducibility

Record for the final results: commit hash of the fork, `study_config.yaml`,
`results/calibration.json`, CPU model, kernel, CPU governor, Python and package
versions (`python -m pip freeze`).
