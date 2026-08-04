STATES = ['High_Demand_Zone', 'Medium_Demand_Zone', 'Low_Demand_Zone']
ACTIONS = ['Match_Passenger', 'Reposition']
GAMMA = 0.9
TRANSITIONS = {
    ('High_Demand_Zone', 'Match_Passenger'): 'High_Demand_Zone',
    ('High_Demand_Zone', 'Reposition'): 'Medium_Demand_Zone',
    ('Medium_Demand_Zone', 'Match_Passenger'): 'Medium_Demand_Zone',
    ('Medium_Demand_Zone', 'Reposition'): 'High_Demand_Zone',
    ('Low_Demand_Zone', 'Match_Passenger'): 'Low_Demand_Zone',
    ('Low_Demand_Zone', 'Reposition'): 'Medium_Demand_Zone',
}
REWARDS = {
    ('High_Demand_Zone', 'Match_Passenger'): 10,
    ('High_Demand_Zone', 'Reposition'): -2,
    ('Medium_Demand_Zone', 'Match_Passenger'): 6,
    ('Medium_Demand_Zone', 'Reposition'): -1,
    ('Low_Demand_Zone', 'Match_Passenger'): 3,
    ('Low_Demand_Zone', 'Reposition'): 2,
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