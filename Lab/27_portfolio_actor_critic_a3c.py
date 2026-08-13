"""
Implement an agent that manages a financial portfolio, choosing stocks to maximize returns and minimize risk using an Actor-Critic (A3C) method to optimize investment.
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

def exp27():
    n_assets = 4
    actor = np.random.randn(n_assets) * 0.1
    critic_w = np.zeros(1)
    returns_mean = np.array([0.04, 0.07, 0.1, 0.02])
    returns_std = np.array([0.01, 0.03, 0.08, 0.005])
    state_feat = np.ones(1)
    for episode in range(1000):
        w = softmax(actor)
        r = np.random.normal(returns_mean, returns_std)
        portfolio_return = np.dot(w, r)
        risk = np.var(r)
        reward = portfolio_return - 0.5 * risk
        value = critic_w @ state_feat
        td_error = reward - value
        critic_w += 0.01 * td_error * state_feat
        actor += 0.01 * td_error * (w * (r - portfolio_return))
    print("exp27", softmax(actor))

if __name__ == "__main__":
    exp27()
