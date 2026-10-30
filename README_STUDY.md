# Deadline-aware planning under CPU contention (fork of DA-MCTS)

This fork extends the code of Gupta, Yuhas and Dubey, *"DA-MCTS: Deadline-Aware Action
Exploration for Safe Real-Time Planning"* (RTCSA 2026). The method, models, maps and the
original experiment code are the authors' work; the authors' README is kept as
`README_UPSTREAM.md`. This README describes only what this fork adds.

## Research question

Can a deadline-aware anytime planner preserve higher trajectory quality and meet real-time
planning deadlines more reliably than a fixed-computation-budget planner when CPU
availability is unpredictable?

The study treats deadline adherence as a first-class metric (not only task completion),
compares a deadline-aware anytime planner against a fixed-work baseline under controlled CPU
contention, and asks whether deadline-aware planning preserves safety and lap completion
under constrained compute.

**Status:** experiments finished (5 seeds per cell, one machine). Full results, tables and
limitations: `docs/RESULTS_AND_LIMITATIONS.md`.

## What this fork adds

| Path | Purpose |
|---|---|
| `run_exps/run_da_mcts_experiments.py` | Modified runner: planner arm selection, per-step timing log, late-command model, per-episode summary CSV |
| `src/controller/da_mcts/controller.py` | Two attributes exposing the last iteration count and budget (applied by `setup/apply_patches.py`) |
| `experiments/study_config.yaml` | All study parameters |
| `experiments/baseline_fixed_budget.py` | One episode of the fixed-work baseline |
| `experiments/adaptive_deadline_aware.py` | One episode of DA-MCTS (or the anytime ablation) |
| `experiments/cpu_contention_experiments.py` | `calibrate`, `run` (resumable sweep), `status` |
| `experiments/cpu_load.py` | Seeded CPU load generator (steady or bursty) |
| `analysis/summarize_results.py` | Per-condition tables and seed-paired comparisons with 95% intervals |
| `analysis/generate_plots.py` | Deadline hit rate, collision rate, lap completion, progress, runtime distribution, trade-off |
| `docs/methodology.md` | The full research protocol (read this first) |
| `docs/paper_notes.md` | Notes on the DA-MCTS paper and related work |
| `docs/CODEBASE_GUIDE.md` | AI-drafted guide to the original code, spot-checked by me |

## Planner arms

- `fixed`: exactly N MCTS iterations per step, no time check (the baseline).
- `anytime`: stops at the time budget, sampling weights stay uniform (ablation).
- `da`: stops at the time budget with deadline-aware sampling weights (the method).

## Setup (Fedora, Python 3.8 environment)

```bash
git clone https://github.com/<your-user>/DA_MCTS_RTCSA_2026.git && cd DA_MCTS_RTCSA_2026
uv venv --seed --python 3.8 venv && source venv/bin/activate
python -m pip install "pip<24.1" "setuptools<66" "wheel<0.40"
python -m pip install gym==0.19.0
python -m pip install --default-timeout=1000 -r requirements.txt
python -m pip install -e f1tenth_gym
python -m pip install -e src/controller/da_mcts/
python -m pip install pandas matplotlib pyyaml
python setup/apply_patches.py
python -c "import da_mcts, f110_gym, torch, gpytorch; print('install OK')"
sudo dnf install -y util-linux kernel-tools
sudo cpupower frequency-set -g performance
```

Always use `python -m pip` inside the environment (a bare `pip` may belong to a different
Python).

## Running the study

```bash
python experiments/cpu_contention_experiments.py calibrate        # idle run, picks N
python experiments/cpu_contention_experiments.py run --seeds 3    # pilot
python experiments/cpu_contention_experiments.py run              # full sweep, resumable
python experiments/cpu_contention_experiments.py status
python analysis/summarize_results.py
python analysis/generate_plots.py
```

Every cell is also run in a **timing-only** variant (late-command model off) for clean
deadline statistics: add `--timing-only` to `run`/`status`, and to the two analysis scripts.
Report deadline statistics from the timing-only variant and crash/completion from the
closed-loop variant (see `docs/methodology.md`, section 5.1).

Use `--dry-run` to see the commands without running anything. Run nothing else heavy on the
machine during the sweep; trials run one at a time on purpose.

## Protocol in one paragraph

Three planner arms are run on the same track and friction under four CPU conditions (none,
light, heavy, bursty), 10 seeds each. The planner is pinned to one core and competing busy
loops are pinned to the same core. Planning time is measured over the whole control step
against the 40 ms control period. A plan that overruns makes the car keep its previous
command (late-command model). Metrics: deadline hit rate, planning-time statistics and
worst case, collision rate, lap completion rate, progress, lap time, steering and
acceleration smoothness, steps before crash. Comparisons are paired by seed with bootstrap
95% intervals. Full details: `docs/methodology.md`.

## Results

With the released DA-MCTS and a 25 ms search budget on a pinned, contended CPU core:

- Contention scaled planning as designed (fixed-work step time 31 ms idle, 65 ms light, 134 ms heavy).
- A time-bounded search did not make the control step meet its 40 ms deadline: whole-step time was
  33 / 43 / 67 ms (idle / light / heavy), because the budget bounds only the search; the work outside it
  grew from about 7.5 ms to about 42 ms.
- With late plans modelled (the car keeps its previous command), every planner crashed in every
  contended closed-loop episode, so that variant cannot rank them under contention.
- Deadline-aware sampling showed no measurable deadline or safety benefit over simply stopping
  the search on time; it drove faster laps (about 49 s vs 75 s at idle).
- Hypotheses H1 (more deadlines met) not supported in practice, H2 not assessable, H3 not testable.

These are 5-seed, single-machine results; see the limitations section before citing them.

## Limitations

Simulation only; one machine; simplified lateness model; wall-clock-dependent planner
(identical seeds do not reproduce identical trajectories); lap times only for completed
episodes; primary study covers one track and one friction value. See `docs/methodology.md`
section 10.

## Attribution, licence and disclosure

Original work: Gupta, Yuhas, Dubey, RTCSA 2026. Check the upstream repository for its licence
before redistributing. The code reading guide (`docs/CODEBASE_GUIDE.md`) were drafted with AI assistance (Claude) and then checked by me; behaviour claims
were verified by running the code where noted.
