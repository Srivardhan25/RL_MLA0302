import numpy as np

GRID = 4
gamma = 0.9

V = np.zeros((GRID, GRID))
policy = [(0,1)]  # always move right

def get_reward(x, y):
    if (x, y) == (3,3):
        return 5
    elif (x, y) == (1,1):
        return -2
    return 0

for _ in range(100):  # iterations
    new_V = np.copy(V)
    for i in range(GRID):
        for j in range(GRID):
            dx, dy = policy[0]
            ni, nj = min(max(i+dx,0),GRID-1), min(max(j+dy,0),GRID-1)
            r = get_reward(ni, nj)
            new_V[i,j] = r + gamma * V[ni,nj]
    V = new_V

print(V)