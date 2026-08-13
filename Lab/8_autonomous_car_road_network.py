"""
Simulate an autonomous car navigating a simple road network with intersections. Design policies for the car to follow traffic rules and reach the destination safely. Implement these policies in Python and evaluate their effectiveness.
"""

import numpy as np
import random

def make_grid(rows, cols, dirt, obstacles, start, gamma=0.9):
    rewards = np.zeros((rows, cols))
    for (r, c) in dirt:
        rewards[r, c] = 1
    for (r, c) in obstacles:
        rewards[r, c] = -1
    return rewards

def valid(r, c, rows, cols, obstacles):
    return 0 <= r < rows and 0 <= c < cols and (r, c) not in obstacles

def step(r, c, a, rows, cols, obstacles):
    dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
    nr, nc = r + dr, c + dc
    if not valid(nr, nc, rows, cols, obstacles):
        return r, c
    return nr, nc

def epsilon_greedy(Q, eps):
    if random.random() < eps:
        return random.randrange(len(Q))
    return int(np.argmax(Q))

def ucb_select(counts, values, t, c=2.0):
    n = len(counts)
    ucb = np.zeros(n)
    for i in range(n):
        if counts[i] == 0:
            return i
        ucb[i] = values[i] + c * np.sqrt(np.log(t + 1) / counts[i])
    return int(np.argmax(ucb))

def thompson_select(alpha, beta):
    samples = [np.random.beta(alpha[i], beta[i]) for i in range(len(alpha))]
    return int(np.argmax(samples))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def exp8():
    rows, cols = 6, 6
    obstacles = [(1, 2), (3, 4), (4, 1)]
    goal = (5, 5)
    intersections = [(2, 2), (3, 3)]
    gamma = 0.9
    V = np.zeros((rows, cols))
    policy = np.random.randint(0, 4, (rows, cols))
    for _ in range(80):
        for r in range(rows):
            for c in range(cols):
                if (r, c) in obstacles:
                    continue
                best = -1e9
                best_a = 0
                for a in range(4):
                    nr, nc = step(r, c, a, rows, cols, obstacles)
                    reward = 10 if (nr, nc) == goal else (-0.5 if (nr, nc) in intersections else -0.1)
                    v = reward + gamma * V[nr, nc]
                    if v > best:
                        best = v
                        best_a = a
                V[r, c] = best
                policy[r, c] = best_a
    r, c = 0, 0
    path = [(r, c)]
    for _ in range(30):
        a = policy[r, c]
        r, c = step(r, c, a, rows, cols, obstacles)
        path.append((r, c))
        if (r, c) == goal:
            break
    print("exp8", path)

if __name__ == "__main__":
    exp8()
