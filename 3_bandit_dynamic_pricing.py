import numpy as np

arms = [0.2, 0.5, 0.8]  # true success probabilities
n_arms = len(arms)
steps = 1000

# Epsilon Greedy
def epsilon_greedy(eps=0.1):
    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)

    for t in range(steps):
        if np.random.rand() < eps:
            a = np.random.randint(n_arms)
        else:
            a = np.argmax(Q)

        reward = 1 if np.random.rand() < arms[a] else 0
        N[a] += 1
        Q[a] += (reward - Q[a]) / N[a]

    return np.sum(Q)

# UCB
def ucb():
    Q = np.zeros(n_arms)
    N = np.ones(n_arms)

    for t in range(steps):
        a = np.argmax(Q + np.sqrt(np.log(t+1)/N))
        reward = 1 if np.random.rand() < arms[a] else 0
        N[a] += 1
        Q[a] += (reward - Q[a]) / N[a]

    return np.sum(Q)

# Thompson Sampling
def thompson():
    alpha = np.ones(n_arms)
    beta = np.ones(n_arms)

    for _ in range(steps):
        samples = np.random.beta(alpha, beta)
        a = np.argmax(samples)
        reward = 1 if np.random.rand() < arms[a] else 0

        if reward:
            alpha[a] += 1
        else:
            beta[a] += 1

    return np.sum(alpha)

print("Epsilon:", epsilon_greedy())
print("UCB:", ucb())
print("Thompson:", thompson())