import numpy as np
import random

GRID_SIZE = 5

# Rewards
grid = np.zeros((GRID_SIZE, GRID_SIZE))
grid[1, 2] = 1   # dirt
grid[3, 3] = 1
grid[2, 1] = -1  # obstacle
grid[4, 0] = -1

actions = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

def step(state, action):
    x, y = state
    dx, dy = action
    nx, ny = x + dx, y + dy

    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
        reward = grid[nx, ny]
        return (nx, ny), reward
    return state, -0.5  # wall penalty

def simulate(policy_steps=20):
    state = (0,0)
    total_reward = 0

    for _ in range(policy_steps):
        action = random.choice(actions)
        state, reward = step(state, action)
        total_reward += reward

    return total_reward

print("Total Reward:", simulate())