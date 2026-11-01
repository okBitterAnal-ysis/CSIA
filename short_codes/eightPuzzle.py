# Experiment 1: 8-Puzzle Problem (A* Search with Manhattan Distance)
import heapq

goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def manhattan(state):
    return sum(abs(i // 3 - (v - 1) // 3) + abs(i % 3 - (v - 1) % 3) for i, v in enumerate(state) if v != 0)

start = tuple(map(int, input("Enter initial 9 tiles separated by space (e.g. 1 2 3 0 4 6 7 5 8): ").split()))

# Priority queue stores: (f_score, g_cost, state, path)
pq = [(manhattan(start), 0, start, [start])]
visited = set([start])

while pq:
    f, g, state, path = heapq.heappop(pq)
    if state == goal:
        print(f"\nSolved in {g} moves:\n")
        for step in path:
            for r in range(0, 9, 3):
                print(step[r:r+3])
            print()
        break

    idx = state.index(0)
    r, c = idx // 3, idx % 3
    moves = []
    if r > 0: moves.append(idx - 3)  # Up
    if r < 2: moves.append(idx + 3)  # Down
    if c > 0: moves.append(idx - 1)  # Left
    if c < 2: moves.append(idx + 1)  # Right

    for next_idx in moves:
        nxt = list(state)
        nxt[idx], nxt[next_idx] = nxt[next_idx], nxt[idx]
        nxt_state = tuple(nxt)
        if nxt_state not in visited:
            visited.add(nxt_state)
            heapq.heappush(pq, (g + 1 + manhattan(nxt_state), g + 1, nxt_state, path + [nxt_state]))
