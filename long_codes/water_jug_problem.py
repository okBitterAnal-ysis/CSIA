#!/usr/bin/env python3
# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 1: State Space Search - Water Jug Problem (BFS)
# =====================================================================

from collections import deque

def solve_water_jug(capacity_a, capacity_b, target):
    """
    Solves the Water Jug Problem using Breadth-First Search (BFS).
    Tracks complete state transitions and execution operations.
    """
    initial_state = (0, 0)
    queue = deque([(initial_state, [("Initial State", initial_state)])])
    visited = set([initial_state])

    print("=" * 60)
    print("WATER JUG PROBLEM SOLVER (BREADTH-FIRST SEARCH)")
    print("=" * 60)
    print(f"Jug 1 Capacity : {capacity_a} liters")
    print(f"Jug 2 Capacity : {capacity_b} liters")
    print(f"Target Volume  : {target} liters")
    print("-" * 60)

    step_count = 0
    while queue:
        (x, y), history = queue.popleft()

        if x == target or y == target:
            print(f"\nTarget achieved! Solution found in {len(history) - 1} steps:")
            print("-" * 60)
            print(f"{'Step':<6} | {'Operation':<30} | {'Jug 1, Jug 2':<15}")
            print("-" * 60)
            for idx, (op, state) in enumerate(history):
                print(f"{idx:<6} | {op:<30} | {str(state):<15}")
            print("=" * 60)
            return history

        # Generate all 6 valid state transitions
        transitions = [
            ("Fill Jug 1 completely", (capacity_a, y)),
            ("Fill Jug 2 completely", (x, capacity_b)),
            ("Empty Jug 1", (0, y)),
            ("Empty Jug 2", (x, 0)),
            ("Pour Jug 1 into Jug 2", (x - min(x, capacity_b - y), y + min(x, capacity_b - y))),
            ("Pour Jug 2 into Jug 1", (x + min(y, capacity_a - x), y - min(y, capacity_a - x)))
        ]

        for op_name, next_state in transitions:
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, history + [(op_name, next_state)]))

    print("\nNo solution exists for the given capacities and target.")
    return None

if __name__ == "__main__":
    jug1_cap = int(input("Enter capacity of Jug 1 (e.g., 4): "))
    jug2_cap = int(input("Enter capacity of Jug 2 (e.g., 3): "))
    target_vol = int(input("Enter target volume (e.g., 2): "))
    solve_water_jug(jug1_cap, jug2_cap, target_vol)
