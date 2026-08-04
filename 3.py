STATES = ['Dirty_Area', 'Clean_Area', 'Near_Obstacle']
ACTIONS = ['Clean', 'Avoid_Obstacle']
GAMMA = 0.9

TRANSITIONS = {
    ('Dirty_Area', 'Clean'): 'Clean_Area',
    ('Dirty_Area', 'Avoid_Obstacle'): 'Near_Obstacle',
    ('Clean_Area', 'Clean'): 'Clean_Area',
    ('Clean_Area', 'Avoid_Obstacle'): 'Near_Obstacle',
    ('Near_Obstacle', 'Clean'): 'Dirty_Area',
    ('Near_Obstacle', 'Avoid_Obstacle'): 'Clean_Area',
}

REWARDS = {
    ('Dirty_Area', 'Clean'): 10,
    ('Dirty_Area', 'Avoid_Obstacle'): -1,
    ('Clean_Area', 'Clean'): 1,
    ('Clean_Area', 'Avoid_Obstacle'): 0,
    ('Near_Obstacle', 'Clean'): -5,
    ('Near_Obstacle', 'Avoid_Obstacle'): 3,
}

V = {s: 0 for s in STATES}
policy = {}

for _ in range(100):
    for s in STATES:
        best_value, best_action = float('-inf'), None
        for a in ACTIONS:
            ns = TRANSITIONS[(s, a)]
            value = REWARDS[(s, a)] + GAMMA * V[ns]
            if value > best_value:
                best_value, best_action = value, a
        V[s] = best_value
        policy[s] = best_action

print("Value Function:", V)
print("Optimal Policy:", policy)