"""
A robot vacuum cleaner navigates a house with various rooms and obstacles. Use the SARSA algorithm to learn the optimal cleaning policy that maximizes the cleaned area while minimizing energy usage. Implement this in Python.
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

def exp12():
    rows, cols = 5, 5
    obstacles = [(1, 1), (2, 2), (3, 3)]
    goal = (4, 4)
    Q = np.zeros((rows, cols, 4))
    alpha = 0.1
    gamma = 0.9
    for episode in range(2000):
        r, c = 0, 0
        a = epsilon_greedy(Q[r, c], 0.2)
        for t in range(100):
            nr, nc = step(r, c, a, rows, cols, obstacles)
            reward = 10 if (nr, nc) == goal else -1
            na = epsilon_greedy(Q[nr, nc], 0.2)
            Q[r, c, a] += alpha * (reward + gamma * Q[nr, nc, na] - Q[r, c, a])
            r, c, a = nr, nc, na
            if (r, c) == goal:
                break
    print("exp12", Q[0, 0])

if __name__ == "__main__":
    exp12()
