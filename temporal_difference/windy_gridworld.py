"""

Sarsa (on-policy Temporal Difference (TD) control) for estimating Q

Example 6.5: Windy Gridworld

    and

Exercise 6.9: Windy Gridworld with King's Moves

    and

Exercise 6.10: Stochastic Wind

"""

import time
from definitions import *

np.random.seed(seed=1234)


X_SIZE = 10
Y_SIZE = 7
START = (0, 3)
GOAL = (7, 3)
WIND = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)
ALPHA = 0.05
GAMMA = 1
R = - 1
STOCHASTIC_WIND = True


def check_boundaries(x, y):
    if x < 0:
        x = 0
    elif x >= X_SIZE:
        x = X_SIZE - 1
    if y < 0:
        y = 0
    elif y >= Y_SIZE:
        y = Y_SIZE - 1
    return x, y


memory_of_environment = dict()


def environment(state, action_id, stochastic=False):
    a = ACTIONS[action_id]
    if stochastic or (state, action_id) not in memory_of_environment:
        x, y = state
        w = WIND[x]
        if stochastic is True and w != 0:
            d = np.random.randint(-1, 2)
            w = w + d
        xp = x + a[0]
        yp = y + a[1] + w
        x, y = check_boundaries(xp, yp)
        if not stochastic:
            memory_of_environment[(state, action_id)] = (x, y)
    if stochastic:
        return x, y
    else:
        return memory_of_environment[(state, action_id)]


def generate_greedy_trajectory(start, policy, stochastic=False):
# def generate_greedy_trajectory(start, q):
    t = []
    state = start
    t.append(state)
    while state != GOAL:
        print(state)
        state = environment(state, policy.lazy_dict[state], stochastic)
        # state = environment(state, action_argmax(q, state))
        t.append(state)
    return t


Q = LazyDict(random_value)
# init terminal state
for a in range(number_of_actions):
    Q[(GOAL, a)] = 0

"""
# random init start state
for a in range(number_of_actions):
    print(Q[(START, a)])
"""

pi = SoftPolicy()

number_of_episodes = 100000

start_time = time.time()

for i in range(number_of_episodes):
    s = START
    pi[s] = action_argmax(Q, s)
    print("ep: ", i)
    a = pi[s]
    while s != GOAL:
        r = R
        sp = environment(s, a, STOCHASTIC_WIND)
        pi[sp] = action_argmax(Q, sp)
        ap = pi[sp]
        Q[(s, a)] = Q[(s, a)] + ALPHA * (r + GAMMA * Q[(sp, ap)] - Q[(s, a)])
        s = sp
        a = ap

print("# --- %s seconds ---" % (time.time() - start_time))
