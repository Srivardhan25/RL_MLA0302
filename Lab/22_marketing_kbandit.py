"""
Simulate a k-armed bandit problem to optimize marketing campaign choices. Implement epsilon-greedy, UCB, and Thompson Sampling algorithms in Python and evaluate their performance.
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

def exp22():
    n_campaigns = 4
    conversion = [0.02, 0.04, 0.01, 0.05]

    def run(strategy, steps=2000):
        Q = np.zeros(n_campaigns)
        counts = np.zeros(n_campaigns)
        alpha = np.ones(n_campaigns)
        beta = np.ones(n_campaigns)
        total = 0
        for t in range(steps):
            if strategy == "eps":
                a = epsilon_greedy(Q, 0.1)
            elif strategy == "ucb":
                a = ucb_select(counts, Q, t)
            else:
                a = thompson_select(alpha, beta)
            r = int(np.random.rand() < conversion[a])
            counts[a] += 1
            Q[a] += (r - Q[a]) / counts[a]
            if strategy == "thompson":
                alpha[a] += r
                beta[a] += 1 - r
            total += r
        return total

    print("exp22", run("eps"), run("ucb"), run("thompson"))

if __name__ == "__main__":
    exp22()
