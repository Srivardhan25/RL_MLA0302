STATES = ['Dry_Soil', 'Moist_Soil', 'Wet_Soil']
ACTIONS = ['Irrigate', 'Skip_Irrigation']
GAMMA = 0.9
TRANSITIONS = {
    ('Dry_Soil', 'Irrigate'): 'Moist_Soil',
    ('Dry_Soil', 'Skip_Irrigation'): 'Dry_Soil',
    ('Moist_Soil', 'Irrigate'): 'Wet_Soil',
    ('Moist_Soil', 'Skip_Irrigation'): 'Moist_Soil',
    ('Wet_Soil', 'Irrigate'): 'Wet_Soil',
    ('Wet_Soil', 'Skip_Irrigation'): 'Moist_Soil',
}
REWARDS = {
    ('Dry_Soil', 'Irrigate'): 8,
    ('Dry_Soil', 'Skip_Irrigation'): -5,
    ('Moist_Soil', 'Irrigate'): 3,
    ('Moist_Soil', 'Skip_Irrigation'): 5,
    ('Wet_Soil', 'Irrigate'): -4,
    ('Wet_Soil', 'Skip_Irrigation'): 6,
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