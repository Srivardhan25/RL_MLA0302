import numpy as np

GRID = 4
gamma = 0.9
V = np.zeros((GRID, GRID))

actions = [(-1,0),(1,0),(0,-1),(0,1)]

def reward(x, y):
    return 10 if (x,y)==(3,3) else -1

# Value Iteration
for _ in range(100):
    new_V = np.copy(V)
    for i in range(GRID):
        for j in range(GRID):
            values = []
            for a in actions:
                ni = min(max(i+a[0],0),GRID-1)
                nj = min(max(j+a[1],0),GRID-1)
                values.append(reward(ni,nj) + gamma * V[ni,nj])
            new_V[i,j] = max(values)
    V = new_V

print(V)