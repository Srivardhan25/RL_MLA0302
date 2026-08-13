"""
In an inventory management system, use Bellman's equation to find the optimal policy for ordering stock. Implement this in Python and demonstrate how the optimal policy minimizes costs.
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

def exp24():
    max_stock = 20
    order_cost = 2
    holding_cost = 1
    shortage_cost = 5
    demand_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    demands = np.arange(len(demand_probs))
    gamma = 0.95
    V = np.zeros(max_stock + 1)
    policy = np.zeros(max_stock + 1, dtype=int)
    for _ in range(200):
        newV = np.copy(V)
        for s in range(max_stock + 1):
            best = 1e9
            best_a = 0
            for a in range(max_stock - s + 1):
                cost = order_cost * a
                expected = 0
                for d, p in zip(demands, demand_probs):
                    sold = min(s + a, d)
                    end_stock = max(s + a - d, 0)
                    shortage = max(d - (s + a), 0)
                    c = cost + holding_cost * end_stock + shortage_cost * shortage
                    expected += p * (c + gamma * V[end_stock])
                if expected < best:
                    best = expected
                    best_a = a
            newV[s] = best
            policy[s] = best_a
        V = newV
    print("exp24", policy)

if __name__ == "__main__":
    exp24()
