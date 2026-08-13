"""
Implement Q-learning to develop an AI agent that plays a simple grid-based game (e.g., a basic version of Pac-Man). The agent should learn to collect rewards (e.g., food) and avoid penalties (e.g., ghosts). Write a Python program to train and evaluate the AI agent.
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

def exp13():
    rows, cols = 5, 5
    ghosts = [(2, 2)]
    food = [(0, 4), (4, 0), (4, 4)]
    Q = np.zeros((rows, cols, 4))
    alpha = 0.1
    gamma = 0.9
    for episode in range(3000):
        r, c = 0, 0
        for t in range(50):
            a = epsilon_greedy(Q[r, c], 0.2)
            nr, nc = step(r, c, a, rows, cols, [])
            reward = 10 if (nr, nc) in food else (-10 if (nr, nc) in ghosts else -0.1)
            Q[r, c, a] += alpha * (reward + gamma * np.max(Q[nr, nc]) - Q[r, c, a])
            r, c = nr, nc
            if (r, c) in ghosts:
                break
    print("exp13", Q[0, 0])

if __name__ == "__main__":
    exp13()
