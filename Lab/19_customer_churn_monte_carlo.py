"""
Use Monte Carlo methods to evaluate a policy for predicting customer churn in a subscription-based service. Implement this policy evaluation in Python and analyze the results.
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

def exp19():
    n_customers = 300
    features = np.random.randn(n_customers, 4)
    true_weights = np.array([0.5, -0.3, 0.2, 0.1])
    churn_prob = 1 / (1 + np.exp(-features @ true_weights))
    policy_weights = np.random.randn(4) * 0.1
    returns = []
    for episode in range(200):
        preds = 1 / (1 + np.exp(-features @ policy_weights))
        reward = -np.mean((preds - churn_prob) ** 2)
        returns.append(reward)
        grad = features.T @ (churn_prob - preds) / n_customers
        policy_weights += 0.1 * grad
    print("exp19", returns[-1])

if __name__ == "__main__":
    exp19()
