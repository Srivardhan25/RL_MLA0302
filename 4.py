STATES = ['Bull_Market', 'Bear_Market', 'Stable_Market']
ACTIONS = ['Buy', 'Sell', 'Hold']
GAMMA = 0.9
TRANSITIONS = {
    ('Bull_Market', 'Buy'): 'Bull_Market',
    ('Bull_Market', 'Sell'): 'Stable_Market',
    ('Bull_Market', 'Hold'): 'Bull_Market',
    ('Bear_Market', 'Buy'): 'Stable_Market',
    ('Bear_Market', 'Sell'): 'Bear_Market',
    ('Bear_Market', 'Hold'): 'Bear_Market',
    ('Stable_Market', 'Buy'): 'Bull_Market',
    ('Stable_Market', 'Sell'): 'Bear_Market',
    ('Stable_Market', 'Hold'): 'Stable_Market',
}
REWARDS = {
    ('Bull_Market', 'Buy'): 8,
    ('Bull_Market', 'Sell'): 5,
    ('Bull_Market', 'Hold'): 6,
    ('Bear_Market', 'Buy'): -5,
    ('Bear_Market', 'Sell'): 4,
    ('Bear_Market', 'Hold'): 1,
    ('Stable_Market', 'Buy'): 3,
    ('Stable_Market', 'Sell'): 2,
    ('Stable_Market', 'Hold'): 4,
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