"""
A delivery robot operates in a warehouse with predefined delivery points. Using Bellman equations, compute the state-value function for navigating to each delivery point. Implement this in Python and visualize the value function for different policies.
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

def exp7():
    rows, cols = 5, 5
    obstacles = [(1, 1), (3, 3)]
    delivery_points = [(0, 4), (4, 0), (2, 2)]
    gamma = 0.9
    policies = []
    for trial in range(3):
        policy = np.random.randint(0, 4, (rows, cols))
        policies.append(policy)
    for policy in policies:
        V = np.zeros((rows, cols))
        for _ in range(60):
            newV = np.zeros((rows, cols))
            for r in range(rows):
                for c in range(cols):
                    if (r, c) in obstacles:
                        continue
                    a = policy[r, c]
                    nr, nc = step(r, c, a, rows, cols, obstacles)
                    reward = 5 if (nr, nc) in delivery_points else -0.1
                    newV[r, c] = reward + gamma * V[nr, nc]
            V = newV
        print("exp7", V)

if __name__ == "__main__":
    exp7()
