""" Example 4.3: Gambler's Problem """

import numpy as np
import math


def action_value_calculation(state, action, v):
    return p_h * v[state + action] + (1-p_h) * v[state - action]


def argmax(agen):
    result, val = None, None
    i = 0
    for a in agen:
        if result:
            if val + margin < a:
                result, val = [i], a
            elif math.fabs(val - a) < margin:
                result.append(i)
        else:
            result, val = [i], a
        i += 1
    return result


# Parameters:

theta = 1.E-32
margin = 1.E-13  # margin in argmax

p_h = 0.25

if __name__ == "__main__":

    # initialize the state-value function V:
    V = np.zeros(101)
    # terminal states:
    # V[0] = 0  # losing
    V[100] = 1  # winning

    i = 0
    while True:
        delta = 0
        for s in range(1, 100):  # np.random.choice(range(1, 100), 99, replace=False)
            v = V[s]
            V[s] = max(action_value_calculation(s, a, V) for a in range(0, min(s, 100-s)+1))
            delta = max(delta, abs(v - V[s]))
        print(i, ": ", delta)
        if delta < theta:
            break
        i += 1

    print(list(V))
    print("\n")

    # find the policy

    pi = [None] * 99
    for s in range(1, 100):
        pi[s-1] = argmax(action_value_calculation(s, a, V) for a in range(0, min(s, 100-s)+1))

    print("Final policy (policies):\n")
    print(pi)
