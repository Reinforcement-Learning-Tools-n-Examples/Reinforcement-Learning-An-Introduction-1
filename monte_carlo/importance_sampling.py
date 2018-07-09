""" Exercise 5.10: Racetrack - Off-policy MC control """

import time
from mc_definitions import *


Q = LazyDict(random_value)
C = LazyDict(const_value(0))

pi = dict()  # target policy
b = SoftPolicy()  # behavior policy (soft policy)

number_of_episodes = 100000000
start_time = time.time()

for i in range(number_of_episodes):
    # b = SoftPolicy()  # behavior policy (soft policy)

    # generate episode using b:
    episode = Episode(b, noise)
    for _ in episode:
        pass

    # print("finish state: ", episode.state_history[-1])
    number_of_steps = len(episode.action_history)

    if i % 1000 == 0:
        print("Episode: ", i)
        print("Ep. length: ", number_of_steps)

    g = 0
    w = 1

    for j in range(number_of_steps-1, -1, -1):
        g = gamma * g - 1
        s = episode.state_history[j]
        a = episode.action_history[j]
        C[(s, a)] = C[(s, a)] + w
        Q[(s, a)] = Q[(s, a)] + w * (g - Q[(s, a)]) / C[(s, a)]
        pi[s] = action_argmax(Q, s)
        # print("state: ", s)
        # print("action: ", a)
        if a != pi[s]:
            if i % 1000 == 0:
                print("update length: ", number_of_steps - j)
                print("action: ", a)
                print("pi[s]: ", pi[s])
            # print(pi)
            break
        w = w / b.probabilities[s][a]

    # The update of behavior policy b significantly speeds-up the learning
    for j in range(number_of_steps - 1, -1, -1):
        s = episode.state_history[j]
        a = episode.action_history[j]
        b[s] = pi[s]
        if a != pi[s]:
            break

print("# --- %s seconds ---" % (time.time() - start_time))
