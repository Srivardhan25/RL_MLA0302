STATES = ['High_Load', 'Medium_Load', 'Low_Load']
ACTIONS = ['Distribute_More', 'Distribute_Less', 'Maintain']
GAMMA = 0.9
TRANSITIONS = {
    ('High_Load', 'Distribute_More'): 'High_Load',
    ('High_Load', 'Distribute_Less'): 'Medium_Load',
    ('High_Load', 'Maintain'): 'High_Load',
    ('Medium_Load', 'Distribute_More'): 'High_Load',
    ('Medium_Load', 'Distribute_Less'): 'Low_Load',
    ('Medium_Load', 'Maintain'): 'Medium_Load',
    ('Low_Load', 'Distribute_More'): 'Medium_Load',
    ('Low_Load', 'Distribute_Less'): 'Low_Load',
    ('Low_Load', 'Maintain'): 'Low_Load',
}
REWARDS = {
    ('High_Load', 'Distribute_More'): -4,
    ('High_Load', 'Distribute_Less'): 6,
    ('High_Load', 'Maintain'): -2,
    ('Medium_Load', 'Distribute_More'): 2,
    ('Medium_Load', 'Distribute_Less'): 3,
    ('Medium_Load', 'Maintain'): 5,
    ('Low_Load', 'Distribute_More'): 4,
    ('Low_Load', 'Distribute_Less'): -1,
    ('Low_Load', 'Maintain'): 2,
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