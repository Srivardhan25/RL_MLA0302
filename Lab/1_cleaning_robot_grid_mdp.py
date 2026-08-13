"""
An autonomous cleaning robot navigates a 5x5 grid where certain cells contain dirt (reward: +1) and obstacles (penalty: -1). The robot starts at the top-left corner and must find an optimal policy to clean the entire grid efficiently. Implement the grid environment as an MDP and write a Python program to simulate the robot's navigation using different policies.
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

def exp1():
    rows, cols = 5, 5
    obstacles = [(1, 1), (2, 3), (3, 0)]
    dirt = [(0, 4), (2, 2), (4, 4), (3, 3)]
    rewards = make_grid(rows, cols, dirt, obstacles, (0, 0))
    r, c = 0, 0
    visited = set()
    path = [(r, c)]
    total = 0
    for _ in range(50):
        actions = list(range(4))
        random.shuffle(actions)
        moved = False
        for a in actions:
            nr, nc = step(r, c, a, rows, cols, obstacles)
            if (nr, nc) != (r, c):
                r, c = nr, nc
                moved = True
                break
        total += rewards[r, c]
        path.append((r, c))
        if len(visited) == rows * cols - len(obstacles):
            break
        visited.add((r, c))
    print("exp1", total, path[-5:])

if __name__ == "__main__":
    exp1()
