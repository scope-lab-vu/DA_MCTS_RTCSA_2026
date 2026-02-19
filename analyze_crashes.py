#!/usr/bin/env python3
"""
Analyze crash trajectories from the MCTS PI-SVGP experiments.
Parses the nohup log to find collision vs done trajectories,
reads the CSVs, and produces a detailed crash analysis.
"""

import re
import os
import numpy as np
from collections import defaultdict

BASE = "/home/samir/Desktop/WORK/f1tenth_gym_ws"
LOG  = os.path.join(BASE, "nohup_svgp_mcts_sw_25_ns_1000_alpha_1.txt")

# ── 1. Parse the log to extract all trajectory info ─────────────────────────

trajectory_info = []  # list of dicts

# We need to match lines like:
# [experiments] INFO: [YasMarina mu=1.0489 t1] Finished at step 3071 (done)
# [src.episode_logger] INFO: Saved 3071-step trajectory -> results/...csv  (reason=done)
# Also capture the ">>> Run N/45: YasMarina  mu=1.0489  task=0  trial=1  seed=42" lines

finished_re = re.compile(
    r'\[(\w+)\s+mu=([\d.]+)\s+t(\d+)\]\s+Finished at step (\d+)\s+\((\w+)\)'
)
saved_re = re.compile(
    r'Saved\s+\d+-step trajectory.*?→\s*(results/\S+\.csv)\s+\(reason=(\w+)\)'
)

with open(LOG, "r") as f:
    lines = f.readlines()

# Parse pairs: Finished line followed by Saved line
pending_finish = None
for line in lines:
    m = finished_re.search(line)
    if m:
        pending_finish = {
            "track": m.group(1),
            "mu": float(m.group(2)),
            "trial": int(m.group(3)),
            "steps": int(m.group(4)),
            "reason": m.group(5),
        }
        continue
    m = saved_re.search(line)
    if m and pending_finish:
        csv_path = os.path.join(BASE, m.group(1))
        pending_finish["csv_path"] = csv_path
        assert pending_finish["reason"] == m.group(2), \
            f"Reason mismatch: {pending_finish['reason']} vs {m.group(2)}"
        trajectory_info.append(pending_finish)
        pending_finish = None

print(f"Parsed {len(trajectory_info)} trajectories from log")
collisions = [t for t in trajectory_info if t["reason"] == "collision"]
successes  = [t for t in trajectory_info if t["reason"] == "done"]
print(f"  Collisions: {len(collisions)}")
print(f"  Successful: {len(successes)}")

# ── 2. Load CSV helper ──────────────────────────────────────────────────────

def load_traj(csv_path):
    """Load trajectory CSV, skip comment lines starting with #."""
    header_lines = 0
    with open(csv_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                header_lines += 1
            else:
                break
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=header_lines,
                         names=True)
    return data

# Column names: S_x, S_y, S_delta, S_vel, S_psi, S_omega, S_beta, A_a, A_sv

# ── 3. Analyze each collision trajectory ─────────────────────────────────────

print("\n" + "="*100)
print("INDIVIDUAL CRASH ANALYSES")
print("="*100)

crash_stats = []  # accumulate for aggregate analysis

for ci, c in enumerate(collisions):
    csv_path = c["csv_path"]
    if not os.path.exists(csv_path):
        print(f"\n  [MISSING] {csv_path}")
        continue

    data = load_traj(csv_path)
    n = len(data)

    # Crash state (last row)
    last = data[-1]
    vel_crash   = last["S_vel"]
    delta_crash = last["S_delta"]
    omega_crash = last["S_omega"]
    beta_crash  = last["S_beta"]
    a_crash     = last["A_a"]
    sv_crash    = last["A_sv"]

    # Last 5 actions before crash
    last5_start = max(0, n - 5)
    last5 = data[last5_start:]

    # Velocity statistics over entire trajectory
    all_vel   = data["S_vel"]
    all_delta = data["S_delta"]
    all_omega = data["S_omega"]
    all_beta  = data["S_beta"]
    all_a     = data["A_a"]
    all_sv    = data["A_sv"]

    # Speed oscillation: compute velocity differences
    vel_diffs = np.diff(all_vel)
    sign_changes = np.sum(np.diff(np.sign(vel_diffs)) != 0)

    # Velocity in last 20 steps
    last20_start = max(0, n - 20)
    vel_last20 = all_vel[last20_start:]
    vel_last20_std = np.std(vel_last20)

    crash_entry = {
        "track": c["track"],
        "mu": c["mu"],
        "trial": c["trial"],
        "steps": c["steps"],
        "vel_crash": vel_crash,
        "delta_crash": delta_crash,
        "omega_crash": omega_crash,
        "beta_crash": beta_crash,
        "vel_max": np.max(all_vel),
        "vel_mean": np.mean(all_vel),
        "delta_max_abs": np.max(np.abs(all_delta)),
        "omega_max_abs": np.max(np.abs(all_omega)),
        "beta_max_abs": np.max(np.abs(all_beta)),
        "vel_std_last20": vel_last20_std,
        "sign_changes": sign_changes,
    }
    crash_stats.append(crash_entry)

    print(f"\n{'─'*100}")
    print(f"CRASH #{ci+1}: {c['track']}  mu={c['mu']}  trial={c['trial']}  "
          f"seed={41+c['trial']}  steps={c['steps']}")
    print(f"  CSV: {csv_path}")
    print(f"  Total steps: {n}")
    print(f"{'─'*100}")

    print(f"\n  [Crash State (step {n})]")
    print(f"    Velocity at crash:  {vel_crash:.4f} m/s")
    print(f"    Steering (delta):   {delta_crash:.6f} rad  ({np.degrees(delta_crash):.2f} deg)")
    print(f"    Yaw rate (omega):   {omega_crash:.6f} rad/s")
    print(f"    Slip angle (beta):  {beta_crash:.6f} rad  ({np.degrees(beta_crash):.2f} deg)")
    print(f"    Action accel (A_a): {a_crash:.4f}")
    print(f"    Action steer vel:   {sv_crash:.4f}")

    print(f"\n  [Last 5 Steps Before Crash]")
    print(f"    {'Step':>6}  {'Vel':>8}  {'Delta':>10}  {'Omega':>10}  {'Beta':>10}  {'A_a':>8}  {'A_sv':>8}")
    for i, row in enumerate(last5):
        step_num = last5_start + i + 1
        print(f"    {step_num:6d}  {row['S_vel']:8.3f}  {row['S_delta']:10.5f}  "
              f"{row['S_omega']:10.5f}  {row['S_beta']:10.5f}  "
              f"{row['A_a']:8.3f}  {row['A_sv']:8.3f}")

    print(f"\n  [Trajectory Statistics]")
    print(f"    Velocity: mean={np.mean(all_vel):.2f}  std={np.std(all_vel):.2f}  "
          f"max={np.max(all_vel):.2f}  min={np.min(all_vel):.2f}")
    print(f"    |Delta|:  max={np.max(np.abs(all_delta)):.5f} ({np.degrees(np.max(np.abs(all_delta))):.2f} deg)  "
          f"mean={np.mean(np.abs(all_delta)):.5f}")
    print(f"    |Omega|:  max={np.max(np.abs(all_omega)):.5f}  mean={np.mean(np.abs(all_omega)):.5f}")
    print(f"    |Beta|:   max={np.max(np.abs(all_beta)):.6f} ({np.degrees(np.max(np.abs(all_beta))):.2f} deg)  "
          f"mean={np.mean(np.abs(all_beta)):.6f}")
    print(f"    Vel oscillations (sign changes in dv): {sign_changes} over {n} steps "
          f"({sign_changes/max(n,1)*100:.1f}%)")
    print(f"    Vel std last 20 steps: {vel_last20_std:.3f}")

# ── 4. Compare with non-collision trajectories (same track/mu) ───────────────

print("\n\n" + "="*100)
print("COMPARISON: COLLISION vs NON-COLLISION (same track/mu)")
print("="*100)

# Group by (track, mu)
groups = defaultdict(lambda: {"collision": [], "done": []})
for t in trajectory_info:
    key = (t["track"], t["mu"])
    groups[key][t["reason"]].append(t)

for (track, mu), group in sorted(groups.items()):
    if not group["collision"]:
        continue  # skip groups without collisions

    print(f"\n{'─'*100}")
    print(f"  {track}  mu={mu}")
    print(f"  Collisions: {len(group['collision'])}  |  Successes: {len(group['done'])}")
    print(f"{'─'*100}")

    # Load all collision trajectories for this group
    col_vels, col_deltas, col_omegas, col_betas = [], [], [], []
    col_crash_vels, col_crash_deltas, col_crash_omegas, col_crash_betas = [], [], [], []
    col_steps = []
    for t in group["collision"]:
        if os.path.exists(t["csv_path"]):
            d = load_traj(t["csv_path"])
            col_vels.extend(d["S_vel"])
            col_deltas.extend(d["S_delta"])
            col_omegas.extend(d["S_omega"])
            col_betas.extend(d["S_beta"])
            col_crash_vels.append(d["S_vel"][-1])
            col_crash_deltas.append(d["S_delta"][-1])
            col_crash_omegas.append(d["S_omega"][-1])
            col_crash_betas.append(d["S_beta"][-1])
            col_steps.append(len(d))

    # Load first non-collision trajectory as reference
    done_vels, done_deltas, done_omegas, done_betas = [], [], [], []
    done_steps = []
    for t in group["done"]:
        if os.path.exists(t["csv_path"]):
            d = load_traj(t["csv_path"])
            done_vels.extend(d["S_vel"])
            done_deltas.extend(d["S_delta"])
            done_omegas.extend(d["S_omega"])
            done_betas.extend(d["S_beta"])
            done_steps.append(len(d))

    col_vels = np.array(col_vels)
    col_deltas = np.array(col_deltas)
    col_omegas = np.array(col_omegas)
    col_betas = np.array(col_betas)

    if len(done_vels) > 0:
        done_vels = np.array(done_vels)
        done_deltas = np.array(done_deltas)
        done_omegas = np.array(done_omegas)
        done_betas = np.array(done_betas)

        print(f"\n    {'Metric':<30}  {'Collision Trajs':>20}  {'Success Trajs':>20}")
        print(f"    {'─'*30}  {'─'*20}  {'─'*20}")
        print(f"    {'Vel max':<30}  {np.max(col_vels):>20.3f}  {np.max(done_vels):>20.3f}")
        print(f"    {'Vel mean':<30}  {np.mean(col_vels):>20.3f}  {np.mean(done_vels):>20.3f}")
        print(f"    {'Vel at crash (mean)':<30}  {np.mean(col_crash_vels):>20.3f}  {'N/A':>20}")
        print(f"    {'|Delta| max (rad)':<30}  {np.max(np.abs(col_deltas)):>20.5f}  {np.max(np.abs(done_deltas)):>20.5f}")
        print(f"    {'|Delta| mean (rad)':<30}  {np.mean(np.abs(col_deltas)):>20.5f}  {np.mean(np.abs(done_deltas)):>20.5f}")
        print(f"    {'|Omega| max':<30}  {np.max(np.abs(col_omegas)):>20.5f}  {np.max(np.abs(done_omegas)):>20.5f}")
        print(f"    {'|Omega| mean':<30}  {np.mean(np.abs(col_omegas)):>20.5f}  {np.mean(np.abs(done_omegas)):>20.5f}")
        print(f"    {'|Beta| max (rad)':<30}  {np.max(np.abs(col_betas)):>20.6f}  {np.max(np.abs(done_betas)):>20.6f}")
        print(f"    {'|Beta| mean (rad)':<30}  {np.mean(np.abs(col_betas)):>20.6f}  {np.mean(np.abs(done_betas)):>20.6f}")
        print(f"    {'Avg episode length':<30}  {np.mean(col_steps):>20.0f}  {np.mean(done_steps):>20.0f}")
    else:
        print("    No successful trajectories for comparison.")


# ── 5. Aggregate crash pattern analysis ──────────────────────────────────────

print("\n\n" + "="*100)
print("AGGREGATE CRASH PATTERN ANALYSIS")
print("="*100)

if crash_stats:
    crash_vels   = [c["vel_crash"] for c in crash_stats]
    crash_deltas = [c["delta_crash"] for c in crash_stats]
    crash_omegas = [c["omega_crash"] for c in crash_stats]
    crash_betas  = [c["beta_crash"] for c in crash_stats]
    crash_steps  = [c["steps"] for c in crash_stats]
    max_vels     = [c["vel_max"] for c in crash_stats]

    print(f"\n  Number of crashes analyzed: {len(crash_stats)}")

    # ── 5a. Speed at crash ───────────────────────────────────────────────
    print(f"\n  [A] VELOCITY AT CRASH")
    print(f"    Mean velocity at crash:   {np.mean(crash_vels):.3f} m/s")
    print(f"    Median velocity at crash: {np.median(crash_vels):.3f} m/s")
    print(f"    Min velocity at crash:    {np.min(crash_vels):.3f} m/s")
    print(f"    Max velocity at crash:    {np.max(crash_vels):.3f} m/s")
    print(f"    Std velocity at crash:    {np.std(crash_vels):.3f} m/s")

    # Categorize: high speed crash vs low speed crash
    # Use median velocity of all successful trajectories as threshold
    all_success_vels = []
    for t in successes:
        if os.path.exists(t["csv_path"]):
            d = load_traj(t["csv_path"])
            all_success_vels.extend(d["S_vel"])
    all_success_vels = np.array(all_success_vels)
    median_success_vel = np.median(all_success_vels)
    mean_success_vel = np.mean(all_success_vels)

    high_speed_crashes = sum(1 for v in crash_vels if v > mean_success_vel)
    low_speed_crashes  = len(crash_vels) - high_speed_crashes
    print(f"\n    Reference: mean success velocity = {mean_success_vel:.3f} m/s")
    print(f"    High-speed crashes (v > mean): {high_speed_crashes}/{len(crash_vels)} ({100*high_speed_crashes/len(crash_vels):.0f}%)")
    print(f"    Low-speed crashes  (v <= mean): {low_speed_crashes}/{len(crash_vels)} ({100*low_speed_crashes/len(crash_vels):.0f}%)")

    # ── 5b. Steering / yaw / slip at crash ───────────────────────────────
    print(f"\n  [B] STEERING / YAW RATE / SLIP ANGLE AT CRASH")
    print(f"    |Delta| at crash:  mean={np.mean(np.abs(crash_deltas)):.5f} rad "
          f"({np.degrees(np.mean(np.abs(crash_deltas))):.2f} deg)  "
          f"max={np.max(np.abs(crash_deltas)):.5f} rad ({np.degrees(np.max(np.abs(crash_deltas))):.2f} deg)")
    print(f"    |Omega| at crash:  mean={np.mean(np.abs(crash_omegas)):.5f}  "
          f"max={np.max(np.abs(crash_omegas)):.5f}")
    print(f"    |Beta| at crash:   mean={np.mean(np.abs(crash_betas)):.6f} rad "
          f"({np.degrees(np.mean(np.abs(crash_betas))):.2f} deg)  "
          f"max={np.max(np.abs(crash_betas)):.6f} rad ({np.degrees(np.max(np.abs(crash_betas))):.2f} deg)")

    # Compare with success trajectory stats
    all_success_deltas = []
    all_success_omegas = []
    all_success_betas  = []
    for t in successes:
        if os.path.exists(t["csv_path"]):
            d = load_traj(t["csv_path"])
            all_success_deltas.extend(d["S_delta"])
            all_success_omegas.extend(d["S_omega"])
            all_success_betas.extend(d["S_beta"])
    all_success_deltas = np.array(all_success_deltas)
    all_success_omegas = np.array(all_success_omegas)
    all_success_betas  = np.array(all_success_betas)

    # Percentile of crash values within success distribution
    print(f"\n    Typical success |delta|: mean={np.mean(np.abs(all_success_deltas)):.5f}  "
          f"p95={np.percentile(np.abs(all_success_deltas), 95):.5f}  "
          f"max={np.max(np.abs(all_success_deltas)):.5f}")
    print(f"    Typical success |omega|: mean={np.mean(np.abs(all_success_omegas)):.5f}  "
          f"p95={np.percentile(np.abs(all_success_omegas), 95):.5f}  "
          f"max={np.max(np.abs(all_success_omegas)):.5f}")
    print(f"    Typical success |beta|:  mean={np.mean(np.abs(all_success_betas)):.6f}  "
          f"p95={np.percentile(np.abs(all_success_betas), 95):.6f}  "
          f"max={np.max(np.abs(all_success_betas)):.6f}")

    # Check if crash values are extreme
    delta_p95 = np.percentile(np.abs(all_success_deltas), 95)
    omega_p95 = np.percentile(np.abs(all_success_omegas), 95)
    beta_p95  = np.percentile(np.abs(all_success_betas), 95)

    extreme_delta = sum(1 for d in crash_deltas if abs(d) > delta_p95)
    extreme_omega = sum(1 for o in crash_omegas if abs(o) > omega_p95)
    extreme_beta  = sum(1 for b in crash_betas if abs(b) > beta_p95)

    print(f"\n    Crashes with |delta| > success p95: {extreme_delta}/{len(crash_stats)} ({100*extreme_delta/len(crash_stats):.0f}%)")
    print(f"    Crashes with |omega| > success p95: {extreme_omega}/{len(crash_stats)} ({100*extreme_omega/len(crash_stats):.0f}%)")
    print(f"    Crashes with |beta|  > success p95: {extreme_beta}/{len(crash_stats)} ({100*extreme_beta/len(crash_stats):.0f}%)")

    # ── 5c. Timing: early vs late crashes ────────────────────────────────
    print(f"\n  [C] TIMING: EARLY vs LATE CRASHES")
    # Get typical episode length for successful runs
    success_steps = [t["steps"] for t in successes]
    mean_success_steps = np.mean(success_steps) if success_steps else 3500

    print(f"    Mean successful episode length: {mean_success_steps:.0f} steps")
    print(f"    Crash step distribution:")
    for c in crash_stats:
        pct = c["steps"] / mean_success_steps * 100
        bar = "#" * int(pct / 2)
        print(f"      {c['track']:12s} mu={c['mu']:.4f} t{c['trial']}: "
              f"step {c['steps']:5d}  ({pct:5.1f}% of avg episode)  {bar}")

    early_crashes = sum(1 for s in crash_steps if s < mean_success_steps * 0.25)
    mid_crashes   = sum(1 for s in crash_steps if mean_success_steps * 0.25 <= s < mean_success_steps * 0.75)
    late_crashes  = sum(1 for s in crash_steps if s >= mean_success_steps * 0.75)

    print(f"\n    Early crashes (<25% of episode): {early_crashes}/{len(crash_stats)} ({100*early_crashes/len(crash_stats):.0f}%)")
    print(f"    Mid crashes (25-75%):            {mid_crashes}/{len(crash_stats)} ({100*mid_crashes/len(crash_stats):.0f}%)")
    print(f"    Late crashes (>75%):             {late_crashes}/{len(crash_stats)} ({100*late_crashes/len(crash_stats):.0f}%)")

    # ── 5d. Velocity oscillation before crash ────────────────────────────
    print(f"\n  [D] VELOCITY OSCILLATION / INSTABILITY BEFORE CRASH")

    for ci, c in enumerate(collisions):
        if not os.path.exists(c["csv_path"]):
            continue
        data = load_traj(c["csv_path"])
        n = len(data)

        # Look at last 50 steps (or all if shorter)
        window = min(50, n)
        last_window = data[-window:]
        vel_w = last_window["S_vel"]

        # Velocity range, std, sign changes in acceleration
        vel_diffs = np.diff(vel_w)
        sign_changes = np.sum(np.diff(np.sign(vel_diffs)) != 0)

        # Check for acceleration reversals (oscillation)
        accel = last_window["A_a"]
        accel_sign_changes = np.sum(np.diff(np.sign(accel)) != 0)

        # Steering velocity reversals
        sv = last_window["A_sv"]
        sv_sign_changes = np.sum(np.diff(np.sign(sv)) != 0)

        print(f"\n    Crash #{ci+1} ({c['track']} mu={c['mu']} t{c['trial']}) "
              f"- last {window} steps:")
        print(f"      Vel range: [{np.min(vel_w):.2f}, {np.max(vel_w):.2f}]  "
              f"std={np.std(vel_w):.3f}  vel_sign_changes={sign_changes}")
        print(f"      Accel sign changes: {accel_sign_changes}  "
              f"Steer-vel sign changes: {sv_sign_changes}")
        print(f"      Mean |A_a|={np.mean(np.abs(accel)):.3f}  "
              f"Mean |A_sv|={np.mean(np.abs(sv)):.3f}")

    # ── 5e. Friction (mu) analysis ───────────────────────────────────────
    print(f"\n  [E] CRASH RATE BY FRICTION COEFFICIENT (mu)")
    mu_crashes = defaultdict(lambda: {"collision": 0, "total": 0})
    for t in trajectory_info:
        mu_crashes[t["mu"]]["total"] += 1
        if t["reason"] == "collision":
            mu_crashes[t["mu"]]["collision"] += 1

    for mu in sorted(mu_crashes.keys()):
        c_count = mu_crashes[mu]["collision"]
        total = mu_crashes[mu]["total"]
        print(f"    mu={mu:.4f}: {c_count}/{total} crashes ({100*c_count/total:.0f}%)")

    # ── 5f. Track analysis ───────────────────────────────────────────────
    print(f"\n  [F] CRASH RATE BY TRACK")
    track_crashes = defaultdict(lambda: {"collision": 0, "total": 0})
    for t in trajectory_info:
        track_crashes[t["track"]]["total"] += 1
        if t["reason"] == "collision":
            track_crashes[t["track"]]["collision"] += 1

    for track in sorted(track_crashes.keys()):
        c_count = track_crashes[track]["collision"]
        total = track_crashes[track]["total"]
        print(f"    {track}: {c_count}/{total} crashes ({100*c_count/total:.0f}%)")

# ── 6. Final summary ────────────────────────────────────────────────────────

print("\n\n" + "="*100)
print("EXECUTIVE SUMMARY")
print("="*100)

print(f"""
Total experiments: {len(trajectory_info)}
Total collisions:  {len(collisions)} ({100*len(collisions)/len(trajectory_info):.0f}%)

KEY FINDINGS:
""")

# Compute key metrics for summary
if crash_stats:
    mean_crash_vel = np.mean(crash_vels)
    median_crash_vel = np.median(crash_vels)

    # Determine if crashes are high-speed
    high_speed_pct = 100 * high_speed_crashes / len(crash_vels)

    # Determine if crashes involve loss of control
    any_extreme = sum(1 for i in range(len(crash_stats))
                      if abs(crash_omegas[i]) > omega_p95 or abs(crash_betas[i]) > beta_p95)
    loss_of_control_pct = 100 * any_extreme / len(crash_stats)

    early_pct = 100 * early_crashes / len(crash_stats)
    mid_pct = 100 * mid_crashes / len(crash_stats)
    late_pct = 100 * late_crashes / len(crash_stats)

    print(f"1. SPEED AT CRASH:")
    print(f"   Mean crash velocity: {mean_crash_vel:.2f} m/s vs mean success velocity: {mean_success_vel:.2f} m/s")
    print(f"   {high_speed_pct:.0f}% of crashes occur at above-average speed")

    print(f"\n2. LOSS OF CONTROL (extreme omega or beta at crash):")
    print(f"   {loss_of_control_pct:.0f}% of crashes show extreme yaw rate or slip angle (>p95 of success)")
    print(f"   {100*extreme_delta/len(crash_stats):.0f}% have extreme steering angle at crash")

    print(f"\n3. CRASH TIMING:")
    print(f"   Early (<25% of episode): {early_pct:.0f}%")
    print(f"   Mid (25-75%):            {mid_pct:.0f}%")
    print(f"   Late (>75%):             {late_pct:.0f}%")

    print(f"\n4. FRICTION EFFECT:")
    for mu in sorted(mu_crashes.keys()):
        c_count = mu_crashes[mu]["collision"]
        total = mu_crashes[mu]["total"]
        print(f"   mu={mu:.4f}: {c_count}/{total} ({100*c_count/total:.0f}%)")

    print(f"\n5. TRACK DIFFICULTY:")
    for track in sorted(track_crashes.keys()):
        c_count = track_crashes[track]["collision"]
        total = track_crashes[track]["total"]
        print(f"   {track}: {c_count}/{total} ({100*c_count/total:.0f}%)")

print("\n" + "="*100)
