STATES = ['Idle', 'Assembling', 'Error_State']
ACTIONS = ['Start_Task', 'Optimize_Process', 'Pause']
GAMMA = 0.9
TRANSITIONS = {
    ('Idle', 'Start_Task'): 'Assembling',
    ('Idle', 'Optimize_Process'): 'Idle',
    ('Idle', 'Pause'): 'Idle',
    ('Assembling', 'Start_Task'): 'Assembling',
    ('Assembling', 'Optimize_Process'): 'Assembling',
    ('Assembling', 'Pause'): 'Idle',
    ('Error_State', 'Start_Task'): 'Error_State',
    ('Error_State', 'Optimize_Process'): 'Idle',
    ('Error_State', 'Pause'): 'Error_State',
}
REWARDS = {
    ('Idle', 'Start_Task'): 5,
    ('Idle', 'Optimize_Process'): 1,
    ('Idle', 'Pause'): 0,
    ('Assembling', 'Start_Task'): 2,
    ('Assembling', 'Optimize_Process'): 8,
    ('Assembling', 'Pause'): -1,
    ('Error_State', 'Start_Task'): -6,
    ('Error_State', 'Optimize_Process'): 4,
    ('Error_State', 'Pause'): -2,
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