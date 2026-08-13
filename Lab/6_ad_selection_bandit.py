"""
An online platform uses bandit algorithms to decide which advertisements to show to users. Implement epsilon-greedy, UCB, and Thompson Sampling algorithms. Use a Python script to determine which algorithm results in the highest click-through rate over time.
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

def exp6():
    n_ads = 4
    ctr = [0.05, 0.08, 0.03, 0.1]

    def pull(i):
        return int(np.random.rand() < ctr[i])

    def run(strategy, steps=2000):
        Q = np.zeros(n_ads)
        counts = np.zeros(n_ads)
        alpha = np.ones(n_ads)
        beta = np.ones(n_ads)
        clicks = 0
        for t in range(steps):
            if strategy == "eps":
                a = epsilon_greedy(Q, 0.1)
            elif strategy == "ucb":
                a = ucb_select(counts, Q, t)
            else:
                a = thompson_select(alpha, beta)
            r = pull(a)
            counts[a] += 1
            Q[a] += (r - Q[a]) / counts[a]
            if strategy == "thompson":
                alpha[a] += r
                beta[a] += 1 - r
            clicks += r
        return clicks

    print("exp6", run("eps"), run("ucb"), run("thompson"))

if __name__ == "__main__":
    exp6()
