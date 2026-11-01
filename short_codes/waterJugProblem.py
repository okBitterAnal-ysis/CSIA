# Experiment 1: Water Jug Problem (State-Space BFS Search)
from collections import deque

c1 = int(input("Enter Jug 1 capacity (e.g. 4): "))
c2 = int(input("Enter Jug 2 capacity (e.g. 3): "))
target = int(input("Enter target amount (e.g. 2): "))

q = deque([(0, 0, [(0, 0)])])
visited = set([(0, 0)])

while q:
    x, y, path = q.popleft()
    if x == target or y == target:
        print(f"\nTarget {target} reached in {len(path) - 1} steps:")
        for step in path:
            print(step)
        break

    next_states = [
        (c1, y),                                  # Fill Jug 1
        (x, c2),                                  # Fill Jug 2
        (0, y),                                   # Empty Jug 1
        (x, 0),                                   # Empty Jug 2
        (x - min(x, c2 - y), y + min(x, c2 - y)), # Pour Jug 1 -> Jug 2
        (x + min(y, c1 - x), y - min(y, c1 - x))  # Pour Jug 2 -> Jug 1
    ]

    for nx, ny in next_states:
        if (nx, ny) not in visited:
            visited.add((nx, ny))
            q.append((nx, ny, path + [(nx, ny)]))
