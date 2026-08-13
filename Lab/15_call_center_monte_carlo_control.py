"""
A call center uses Monte Carlo methods to optimize the assignment of customer service representatives to incoming calls. Implement Monte Carlo policy control in Python to minimize average call handling time.
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

def exp15():
    n_reps = 4
    n_calls = 100
    Q = np.zeros((n_calls, n_reps))
    counts = np.zeros((n_calls, n_reps))
    handle_time = np.random.exponential(5, (n_calls, n_reps))
    for episode in range(500):
        assignment = [epsilon_greedy(Q[call], 0.2) for call in range(n_calls)]
        total_return = sum(-handle_time[call, assignment[call]] for call in range(n_calls))
        for call in range(n_calls):
            counts[call, assignment[call]] += 1
            Q[call, assignment[call]] += (total_return - Q[call, assignment[call]]) / counts[call, assignment[call]]
    print("exp15", Q.mean())

if __name__ == "__main__":
    exp15()
