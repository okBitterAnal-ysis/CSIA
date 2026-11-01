# =====================================================================
# BHARATIYA VIDYA BHAVAN'S SARDAR PATEL INSTITUTE OF TECHNOLOGY
# Experiment 1: Heuristic Search - 8-Puzzle Problem (A* Algorithm)
# =====================================================================

import heapq

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def manhattan_distance(state):
    """Calculates total Manhattan Distance heuristic."""
    distance = 0
    for idx, val in enumerate(state):
        if val != 0:
            target_idx = val - 1
            distance += abs(idx // 3 - target_idx // 3) + abs(idx % 3 - target_idx % 3)
    return distance

def misplaced_tiles(state):
    """Calculates number of misplaced tiles heuristic."""
    return sum(1 for idx, val in enumerate(state) if val != 0 and val != GOAL_STATE[idx])

def print_board(state):
    """Prints a 3x3 board formatted nicely."""
    for r in range(0, 9, 3):
        print(f"  {state[r]} {state[r+1]} {state[r+2]}")

def solve_eight_puzzle(start_state):
    """Solves the 8-puzzle using A* search with Manhattan Distance heuristic."""
    print("=" * 60)
    print("8-PUZZLE PROBLEM SOLVER (A* HEURISTIC SEARCH)")
    print("=" * 60)
    print("Initial State:")
    print_board(start_state)
    print(f"\nInitial Manhattan Distance : {manhattan_distance(start_state)}")
    print(f"Initial Misplaced Tiles    : {misplaced_tiles(start_state)}")
    print("-" * 60)

    # Priority queue stores: (f_score, g_cost, state, path)
    pq = [(manhattan_distance(start_state), 0, start_state, [start_state])]
    visited = set([start_state])
    nodes_expanded = 0

    while pq:
        f, g, state, path = heapq.heappop(pq)
        nodes_expanded += 1

        if state == GOAL_STATE:
            print(f"\nGoal state reached successfully in {g} moves!")
            print(f"Total nodes expanded: {nodes_expanded}")
            print("-" * 60)
            for step_num, step_state in enumerate(path):
                print(f"Step {step_num} (g = {step_num}, h = {manhattan_distance(step_state)}):")
                print_board(step_state)
                print()
            print("=" * 60)
            return path

        idx = state.index(0)
        r, c = idx // 3, idx % 3
        possible_moves = []
        if r > 0: possible_moves.append(("UP", idx - 3))
        if r < 2: possible_moves.append(("DOWN", idx + 3))
        if c > 0: possible_moves.append(("LEFT", idx - 1))
        if c < 2: possible_moves.append(("RIGHT", idx + 1))

        for direction, next_idx in possible_moves:
            new_board = list(state)
            new_board[idx], new_board[next_idx] = new_board[next_idx], new_board[idx]
            next_state = tuple(new_board)

            if next_state not in visited:
                visited.add(next_state)
                h = manhattan_distance(next_state)
                heapq.heappush(pq, (g + 1 + h, g + 1, next_state, path + [next_state]))

    print("No solution could be found.")
    return None

if __name__ == "__main__":
    raw_input = input("Enter 9 space-separated puzzle tiles (0 for blank, e.g., 1 2 3 0 4 6 7 5 8): ").split()
    initial_tiles = tuple(map(int, raw_input))
    solve_eight_puzzle(initial_tiles)
