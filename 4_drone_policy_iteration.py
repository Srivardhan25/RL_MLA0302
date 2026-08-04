import numpy as np

gamma = 0.9
GRID = 4

V = np.zeros((GRID, GRID))
policy = np.zeros((GRID, GRID, 2), dtype=int)

actions = [(-1,0),(1,0),(0,-1),(0,1)]

def reward(x, y):
    if (x, y) == (3,3):
        return 10
    return -1

for _ in range(10):

    # Policy Evaluation
    for _ in range(50):
        for i in range(GRID):
            for j in range(GRID):
                dx, dy = policy[i, j]
                ni = int(min(max(i + dx, 0), GRID - 1))
                nj = int(min(max(j + dy, 0), GRID - 1))
                V[i, j] = reward(ni, nj) + gamma * V[ni, nj]

    # Policy Improvement
    for i in range(GRID):
        for j in range(GRID):
            best = float("-inf")
            best_action = (0, 0)

            for dx, dy in actions:
                ni = min(max(i + dx, 0), GRID - 1)
                nj = min(max(j + dy, 0), GRID - 1)

                value = reward(ni, nj) + gamma * V[ni, nj]

                if value > best:
                    best = value
                    best_action = (dx, dy)

            policy[i, j] = best_action

print("Value Function:")
print(V)

print("\nOptimal Policy:")
print(policy)