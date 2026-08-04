STATES = ['High_CTR_Segment', 'Medium_CTR_Segment', 'Low_CTR_Segment']
ACTIONS = ['Show_Ad_A', 'Show_Ad_B']
GAMMA = 0.9
TRANSITIONS = {
    ('High_CTR_Segment', 'Show_Ad_A'): 'High_CTR_Segment',
    ('High_CTR_Segment', 'Show_Ad_B'): 'Medium_CTR_Segment',
    ('Medium_CTR_Segment', 'Show_Ad_A'): 'High_CTR_Segment',
    ('Medium_CTR_Segment', 'Show_Ad_B'): 'Medium_CTR_Segment',
    ('Low_CTR_Segment', 'Show_Ad_A'): 'Medium_CTR_Segment',
    ('Low_CTR_Segment', 'Show_Ad_B'): 'Low_CTR_Segment',
}
REWARDS = {
    ('High_CTR_Segment', 'Show_Ad_A'): 9,
    ('High_CTR_Segment', 'Show_Ad_B'): 4,
    ('Medium_CTR_Segment', 'Show_Ad_A'): 6,
    ('Medium_CTR_Segment', 'Show_Ad_B'): 5,
    ('Low_CTR_Segment', 'Show_Ad_A'): 3,
    ('Low_CTR_Segment', 'Show_Ad_B'): 1,
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
        policy[s] = best_actio
print("Value Function:", V)
print("Optimal Policy:", policy)