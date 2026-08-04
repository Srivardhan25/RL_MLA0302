STATES = ['Beginner_Level', 'Intermediate_Level', 'Advanced_Level']
ACTIONS = ['Recommend_Familiar_Content', 'Recommend_New_Content']
GAMMA = 0.9
TRANSITIONS = {
    ('Beginner_Level', 'Recommend_Familiar_Content'): 'Beginner_Level',
    ('Beginner_Level', 'Recommend_New_Content'): 'Intermediate_Level',
    ('Intermediate_Level', 'Recommend_Familiar_Content'): 'Intermediate_Level',
    ('Intermediate_Level', 'Recommend_New_Content'): 'Advanced_Level',
    ('Advanced_Level', 'Recommend_Familiar_Content'): 'Advanced_Level',
    ('Advanced_Level', 'Recommend_New_Content'): 'Advanced_Level',
}
REWARDS = {
    ('Beginner_Level', 'Recommend_Familiar_Content'): 4,
    ('Beginner_Level', 'Recommend_New_Content'): 6,
    ('Intermediate_Level', 'Recommend_Familiar_Content'): 5,
    ('Intermediate_Level', 'Recommend_New_Content'): 8,
    ('Advanced_Level', 'Recommend_Familiar_Content'): 6,
    ('Advanced_Level', 'Recommend_New_Content'): 7,
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