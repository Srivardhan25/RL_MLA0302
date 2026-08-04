STATES = ['Known_Route', 'Unknown_Route', 'Congested_Route']
ACTIONS = ['Explore_New_Path', 'Use_Known_Path']
GAMMA = 0.9
TRANSITIONS = {
    ('Known_Route', 'Explore_New_Path'): 'Unknown_Route',
    ('Known_Route', 'Use_Known_Path'): 'Known_Route',
    ('Unknown_Route', 'Explore_New_Path'): 'Unknown_Route',
    ('Unknown_Route', 'Use_Known_Path'): 'Known_Route',
    ('Congested_Route', 'Explore_New_Path'): 'Unknown_Route',
    ('Congested_Route', 'Use_Known_Path'): 'Congested_Route',
}
REWARDS = {
    ('Known_Route', 'Explore_New_Path'): 3,
    ('Known_Route', 'Use_Known_Path'): 6,
    ('Unknown_Route', 'Explore_New_Path'): 5,
    ('Unknown_Route', 'Use_Known_Path'): 2,
    ('Congested_Route', 'Explore_New_Path'): 7,
    ('Congested_Route', 'Use_Known_Path'): -4,
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