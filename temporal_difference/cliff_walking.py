"""

Sarsa and Q-learning (on-policy Temporal Difference (TD) control) for estimating Q

Example 6.6: Cliff Walking

"""

import time
from definitions import *


np.random.seed(seed=1234)


number_of_episodes = 20000

X_SIZE = 12
Y_SIZE = 4
START = (0, 0)
GOAL = (X_SIZE-1, 0)
ALPHA = 0.5
GAMMA = 1
R1 = - 1
R2 = -100

CLIFF = tuple((i, 0) for i in range(1, X_SIZE-1))  # Any kind of obstacle can be defined here as a "cliff"


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


def environment(state, action_id):
    a = ACTIONS[action_id]
    if (state, action_id) not in memory_of_environment:
        x, y = state
        xp = x + a[0]
        yp = y + a[1]
        if (xp, yp) in CLIFF:
            x, y = START
            r = R2
        else:
            x, y = check_boundaries(xp, yp)
            r = R1
        memory_of_environment[(state, action_id)] = ((x, y), r)
    return memory_of_environment[(state, action_id)]


def generate_greedy_trajectory(start, policy):
    t = []
    state = start
    t.append(state)
    while state != GOAL:
        print(state)
        state, _ = environment(state, policy.lazy_dict[state])
        t.append(state)
    return t


# 1) Sarsa (on-policy learning):

QS = LazyDict(random_value)
# init terminal state
for a in range(number_of_actions):
    QS[(GOAL, a)] = 0

pi_sarsa = SoftPolicy()

start_time = time.time()

for i in range(number_of_episodes):
    print("episode: ", i)
    s = START
    pi_sarsa[s] = action_argmax(QS, s)
    a = pi_sarsa[s]
    while s != GOAL:
        sp, r = environment(s, a)
        pi_sarsa[sp] = action_argmax(QS, sp)
        ap = pi_sarsa[sp]
        QS[(s, a)] = QS[(s, a)] + ALPHA * (r + GAMMA * QS[(sp, ap)] - QS[(s, a)])
        s = sp
        a = ap

print("# --- %s seconds ---" % (time.time() - start_time))

# 2) Q-learning (off-policy learning):

QQL = LazyDict(random_value)
# init terminal state
for a in range(number_of_actions):
    QQL[(GOAL, a)] = 0

pi_ql = SoftPolicy()

for i in range(number_of_episodes):
    print("episode: ", i)
    s = START
    while s != GOAL:
        pi_ql[s] = action_argmax(QQL, s)
        a = pi_ql[s]
        sp, r = environment(s, a)
        maxa = action_argmax(QQL, sp)
        QQL[(s, a)] = QQL[(s, a)] + ALPHA * (r + GAMMA * QQL[(sp, maxa)] - QQL[(s, a)])
        s = sp

print("# --- %s seconds ---" % (time.time() - start_time))
