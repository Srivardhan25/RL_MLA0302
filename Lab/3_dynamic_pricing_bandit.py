"""
An online retailer uses a multi-armed bandit approach to set prices dynamically. Simulate different pricing strategies using epsilon-greedy, UCB, and Thompson Sampling. Write a Python script to compare which strategy maximizes revenue over a series of pricing decisions.
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

def exp3():
    prices = [10, 12, 15, 18, 20]
    n = len(prices)
    demand_prob = [0.9, 0.8, 0.6, 0.4, 0.2]

    def pull(i):
        sold = np.random.rand() < demand_prob[i]
        return prices[i] * sold

    def epsilon_greedy_run(steps=1000, eps=0.1):
        Q = np.zeros(n)
        counts = np.zeros(n)
        total = 0
        for t in range(steps):
            a = epsilon_greedy(Q, eps)
            r = pull(a)
            counts[a] += 1
            Q[a] += (r - Q[a]) / counts[a]
            total += r
        return total

    def ucb_run(steps=1000):
        values = np.zeros(n)
        counts = np.zeros(n)
        total = 0
        for t in range(steps):
            a = ucb_select(counts, values, t)
            r = pull(a)
            counts[a] += 1
            values[a] += (r - values[a]) / counts[a]
            total += r
        return total

    def thompson_run(steps=1000):
        alpha = np.ones(n)
        beta = np.ones(n)
        total = 0
        for t in range(steps):
            a = thompson_select(alpha, beta)
            r = pull(a)
            success = r > 0
            alpha[a] += success
            beta[a] += 1 - success
            total += r
        return total

    print("exp3", epsilon_greedy_run(), ucb_run(), thompson_run())

if __name__ == "__main__":
    exp3()
