"""
A delivery drone needs to find the shortest path from a warehouse to multiple delivery points in a city represented as a grid. Implement a policy iteration algorithm using dynamic programming to find the optimal route policy in Python.
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

def exp4():
    rows, cols = 6, 6
    obstacles = [(2, 2), (2, 3), (3, 2)]
    delivery_points = [(0, 5), (5, 0), (5, 5)]
    gamma = 0.9
    policy = np.random.randint(0, 4, (rows, cols))
    V = np.zeros((rows, cols))
    for _ in range(50):
        for r in range(rows):
            for c in range(cols):
                if (r, c) in obstacles:
                    continue
                best_v = -1e9
                best_a = 0
                for a in range(4):
                    nr, nc = step(r, c, a, rows, cols, obstacles)
                    reward = 5 if (nr, nc) in delivery_points else -0.1
                    v = reward + gamma * V[nr, nc]
                    if v > best_v:
                        best_v = v
                        best_a = a
                policy[r, c] = best_a
                V[r, c] = best_v
    print("exp4", V)

if __name__ == "__main__":
    exp4()
