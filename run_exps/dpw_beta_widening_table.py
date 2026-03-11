"""
DPW Outcome Progressive Widening: Children vs Visits
=====================================================
Simulates how many distinct outcome children a chance node accumulates
as a function of visit count N and widening exponent beta.

Widening condition: add new child when  children_count <= N^beta
(i.e. target children = floor(N^beta), capped at 1 new child per visit)
"""

import math

BETAS = [0.05, 0.1, 0.15, 0.175, 0.2]
VISITS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Annotate visits relevant to a 35ms budget (~300-500 total iters)
BUDGET_RANGE = (50, 200)  # realistic visits to a near-root chance node


def simulate_children(max_visits: int, beta: float) -> dict:
    """Simulate DPW widening: how many children after N visits."""
    k = 0
    results = {}
    for n in range(1, max_visits + 1):
        if k <= n ** beta:
            k += 1
        if n in VISITS:
            results[n] = k
    return results


def main():
    # Build table: rows = N (visits), cols = beta
    table = {}
    for beta in BETAS:
        table[beta] = simulate_children(max(VISITS), beta)

    # Header
    col_w = 8
    beta_header = "".join(f"  b={b:<5}" for b in BETAS)
    sep = "-" * (10 + len(beta_header))

    print("\nDPW Outcome Widening: Number of distinct children at a chance node")
    print(f"  Condition: add new child when children_count <= N^beta")
    print(f"  Budget regime (shaded): N ∈ [{BUDGET_RANGE[0]}, {BUDGET_RANGE[1]}] visits\n")
    print(f"{'N (visits)':<10}{beta_header}")
    print(sep)

    for n in VISITS:
        in_budget = BUDGET_RANGE[0] <= n <= BUDGET_RANGE[1]
        marker = " <<" if in_budget else ""
        row = "".join(f"  {table[b][n]:<7}" for b in BETAS)
        print(f"{n:<10}{row}{marker}")

    print(sep)
    print("\nColumns: beta (widening exponent)")
    print("<<  = realistic visit count for near-root chance node in 35ms budget\n")

    # Inverted table: rows = beta, cols = k children
    K_TARGETS = [1, 2, 3, 4, 5, 10]
    col_w = 12

    print("\nVisits needed to reach k distinct children:")
    header = f"  {'beta':<8}" + "".join(f"{'k='+str(k):>{col_w}}" for k in K_TARGETS)
    sep2 = "-" * len(header)
    print(header)
    print(sep2)

    for beta in BETAS:
        row = f"  {beta:<8}"
        for k_target in K_TARGETS:
            if k_target == 1:
                n_needed = 1
            else:
                n_needed = math.ceil(k_target ** (1.0 / beta))
            if n_needed <= 3000:
                cell = f"{n_needed:,}"
            else:
                cell = f"{n_needed:.1e}*"
            row += f"{cell:>{col_w}}"
        print(row)

    print(sep2)
    print("  * = unreachable within ~3000 iterations (35ms budget)")


if __name__ == "__main__":
    main()
