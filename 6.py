STATES = ['Low_Vitals', 'Normal_Vitals', 'High_Risk_Vitals']
ACTIONS = ['Increase_Dosage', 'Decrease_Dosage', 'Maintain_Dosage']
GAMMA = 0.9
TRANSITIONS = {
    ('Low_Vitals', 'Increase_Dosage'): 'Normal_Vitals',
    ('Low_Vitals', 'Decrease_Dosage'): 'Low_Vitals',
    ('Low_Vitals', 'Maintain_Dosage'): 'Low_Vitals',
    ('Normal_Vitals', 'Increase_Dosage'): 'High_Risk_Vitals',
    ('Normal_Vitals', 'Decrease_Dosage'): 'Low_Vitals',
    ('Normal_Vitals', 'Maintain_Dosage'): 'Normal_Vitals',
    ('High_Risk_Vitals', 'Increase_Dosage'): 'High_Risk_Vitals',
    ('High_Risk_Vitals', 'Decrease_Dosage'): 'Normal_Vitals',
    ('High_Risk_Vitals', 'Maintain_Dosage'): 'High_Risk_Vitals',
}
REWARDS = {
    ('Low_Vitals', 'Increase_Dosage'): 8,
    ('Low_Vitals', 'Decrease_Dosage'): -2,
    ('Low_Vitals', 'Maintain_Dosage'): 1,
    ('Normal_Vitals', 'Increase_Dosage'): -3,
    ('Normal_Vitals', 'Decrease_Dosage'): 2,
    ('Normal_Vitals', 'Maintain_Dosage'): 9,
    ('High_Risk_Vitals', 'Increase_Dosage'): -10,
    ('High_Risk_Vitals', 'Decrease_Dosage'): 6,
    ('High_Risk_Vitals', 'Maintain_Dosage'): -4,
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