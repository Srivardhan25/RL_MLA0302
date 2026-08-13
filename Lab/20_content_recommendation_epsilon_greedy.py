"""
Implement an epsilon-greedy strategy to optimize content recommendations on an online learning platform. Write a Python script to simulate and analyze its performance over multiple runs.
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

def exp20():
    n_content = 6
    engagement = [0.2, 0.5, 0.3, 0.7, 0.1, 0.4]
    Q = np.zeros(n_content)
    counts = np.zeros(n_content)
    history = []
    for t in range(1500):
        a = epsilon_greedy(Q, 0.15)
        r = np.random.rand() < engagement[a]
        counts[a] += 1
        Q[a] += (r - Q[a]) / counts[a]
        history.append(r)
    print("exp20", np.mean(history[-200:]))

if __name__ == "__main__":
    exp20()
