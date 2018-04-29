""" Example 4.2: Jack's Car Rental """

import numpy as np
import math
import sys


def poisson(lam, n):
    """Return probability of getting n for Poisson distribution with expected value lam"""
    return math.pow(lam, n) * math.exp(-lam) / math.factorial(n)


def poisson_cumulative(lam, n):
    """Return cumulative probability for the Poisson distribution"""
    return sum(poisson(lam, i) for i in range(n+1))


def available_actions_range(state):
    s1, s2 = state
    upper = min(s1, max_move)
    lower = - min(s2, max_move)
    return range(lower, upper+1)


def renting(s, lam, req):
    prob = poisson(lam, req) if req < (s + 1) else (1 - poisson_cumulative(lam, s))
    rented = req if req < (s + 1) else s
    return prob, rented


def returning(s, lam, ret):
    prob = poisson(lam, ret) if ret < (size - s + 1) else (1 - poisson_cumulative(lam, size - s))
    returned = ret if ret < (size - s + 1) else (size - s)
    return prob, returned


def cost_modification(action, state):
    """Return the cost change to solve Exercise 4.5"""
    s1, s2 = state
    cost_change = 0
    if action > 0:
        cost_change -= cost_per_car_moved * 1  # one free ride from loc 1 to loc 2
    capacity = 10
    parking_cost_per_additional_car = 4
    if s1 > capacity:
        cost_change += parking_cost_per_additional_car * (s1 - capacity)
    if s2 > capacity:
        cost_change += parking_cost_per_additional_car * (s2 - capacity)
    return cost_change


def p_build_function(state, action):
    """Return the conditional probabilities p(s', r | state, action) as a dictionary for fixed state and action

    Note: action has to be meaningful (e.g., an element from function available_actions_range

    """

    s1, s2 = state
    s1 = (s1 - action) if (s1 - action) <= size else size
    s2 = (s2 + action) if (s2 + action) <= size else size

    cost = cost_per_car_moved * abs(action)

    # Exercise 4.5 (comment-out if replicating the results from the Example 4.2)
    cost += cost_modification(action, state)

    result = dict()
    for req1 in range(0, s1+2):
        prob_req1, rented1 = renting(s1, lam_req1, req1)
        for req2 in range(0, s2+2):
            prob_req2, rented2 = renting(s2, lam_req2, req2)
            r = credit_per_car_rented * (rented1 + rented2) - cost
            prob_req = prob_req1 * prob_req2
            for ret1 in range(0, size - s1 + 2):
                prob_ret1, returned1 = returning(s1, lam_ret1, ret1)
                sp1 = s1 - rented1 + returned1
                for ret2 in range(0, size - s2 + 2):
                    prob_ret2, returned2 = returning(s2, lam_ret2, ret2)
                    sp2 = s2 - rented2 + returned2
                    sp = (sp1, sp2)
                    prob = prob_req * prob_ret1 * prob_ret2
                    if sp in result:
                        if r in result[sp]:
                            result[sp][r] = result[sp][r] + prob
                        else:
                            result[sp][r] = prob
                    else:
                        result[sp] = {r: prob}
    return result


def action_value_calculation(state, action, v):
    p_sa = p[(state, action)]
    v_sa = 0
    for sp in p_sa.keys():
        for r in p_sa[sp]:
            v_sa += p_sa[sp][r] * (r + gamma * v[sp])
    return v_sa


def state_value_function_update_rule(state, policy, v):
    action = policy[state]
    vp = action_value_calculation(state, action, v)
    return vp


def states():
    for s in range((size+1)*(size+1)):
        # yield (s // (size+1), s % (size+1))
        yield divmod(s, size+1)


def policy_evaluation(V, policy):
    """Policy evaluation for deterministic policy"""
    i = 0
    while True:
        delta = 0
        for state in states():
            v = V[state]
            V[state] = state_value_function_update_rule(state, policy, V)
            delta = max(delta, abs(v - V[state]))
        print(i, ": ",  delta)
        if delta < theta:
            break
        i += 1
    return V


def policy_improvement(V, policy):
    stable = True
    for state in states():
        old_action = policy[state]
        a_max = old_action
        v_sa = action_value_calculation(state, old_action, V)
        for a in available_actions_range(state):
            if v_sa < action_value_calculation(state, a, V):
                v_sa = action_value_calculation(state, a, V)
                a_max = a
        if old_action != a_max:
            policy[state] = a_max
            stable = False
    return stable


def grid_to_string(size, x):
    result = ''
    for s1 in range(0, size + 1):
        for s2 in range(0, size + 1):
            result += str(x[(s1, s2)]) + "\t"
        result += "\n"
    return result


def test_prob_dist_property(p):
    print("Testing if sum_{s',r} p(s',r|s, a)==1 for all s and a (this might take some time)\n")
    for s1 in range(0, size + 1):
        for s2 in range(0, size + 1):
            s = (s1, s2)
            for a in available_actions_range(s):
                list_of_r_to_p_dict = p[(s, a)].values()
                list_of_partial_sums = []
                for item in list_of_r_to_p_dict:
                    list_of_partial_sums.append(sum(p for p in item.values()))
                x = abs(sum(list_of_partial_sums))
                if abs(1 - x) > 1.E-14:
                    print("sum_{s', r} p(s', r| (%d, %d), %d)=%.16f which is not 1!" % (s1, s2, a, x))
                    sys.exit(1)
    print("Test passed!\n")


# Parameters setting:

size = 10  # maximal number of cars at each location
max_move = 5  # maximal number of cars we can move at one night
cost_per_car_moved = 2
credit_per_car_rented = 10

lam_req1 = 3
lam_req2 = 4
lam_ret1 = 3
lam_ret2 = 2

gamma = 0.9  # discount factor
theta = 1.E-10  # convergence threshold

if __name__ == "__main__":

    # 1. Initialization

    # initialize the state-value function V:
    V = dict()
    for state in states():
        V[state] = np.random.rand()

    # initialize the policy (with no cars moving for all states, i.e., a=0)
    policy = dict()
    for state in states():
        policy[state] = 0

    init_policy = grid_to_string(size, policy)
    print("Initial policy: \n")
    print(init_policy)
    f = open('policy_%d.dat' % 0, 'w')
    f.write(init_policy)
    f.close()

    print("Saving the environment dynamics (this might take some time)...\n")
    p = dict()
    for state in states():
        for a in available_actions_range(state):
            p[(state, a)] = p_build_function(state, a)

    # test_prob_dist_property(p)  # comment out this line after testing

    # Policy iteration:
    k = 0
    while True:
        # 2. Policy evaluation (prediction)
        V_old = V.copy()
        policy_evaluation(V, policy)

        # checking if V has changed (according to the Exercise 4.4):
        if V_old == V:
            print("State-value function V has not changed.\n")
            break

        # 3. Policy improvement
        policy_stable = policy_improvement(V, policy)
        if policy_stable:
            print("Policy has not changed.\n")
            break
        k += 1

        # visualization of the policy
        kth_policy = grid_to_string(size, policy)
        print(kth_policy)
        f = open('policy_%d.dat' % k, 'w')
        f.write(kth_policy)
        f.close()

    # value function visualization:
    final_vals = grid_to_string(size, V)
    print(final_vals)
    f = open('final_value_func.dat', 'w')
    f.write(final_vals)
    f.close()
