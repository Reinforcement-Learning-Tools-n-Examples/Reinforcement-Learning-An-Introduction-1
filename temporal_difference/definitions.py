
import numpy as np
from actions import ACTIONS


number_of_actions = len(ACTIONS)
IDs = list(range(0, number_of_actions))
EPS = 0.1


def get_random_action_id():
    return np.random.randint(0, number_of_actions)


def get_eps_soft_distribution(a):
    # print("a: ", a)
    return [EPS / number_of_actions] * a + \
           [1 - EPS + EPS / number_of_actions] + \
           [EPS / number_of_actions] * (number_of_actions - a - 1)


def random_value():
    return np.random.rand()


def const_value(const=0):
    return lambda: const


class LazyDict(object):

    def __init__(self, method=None):
        self.data = dict()
        self.method = method

    def insert(self, key, value):
        self.data[key] = value

    def search(self, key):
        if key not in self.data:
            self.data[key] = self.method()
        return self.data[key]

    def __getitem__(self, key):
        """Square bracket [] accessor for getting the item with key."""
        return self.search(key)

    def __setitem__(self, key, value):
        """Square bracket [] accessor for setting the key-value item."""
        self.insert(key, value)


def action_argmax(q, state):
    val = None
    idx = None
    for action_id in range(0, number_of_actions):
        if val is None:
            val = q[(state, action_id)]
            idx = action_id
            continue
        if val < q[(state, action_id)]:
            val = q[(state, action_id)]
            idx = action_id
    return idx


class SoftPolicy(object):

    def __init__(self):
        self.lazy_dict = LazyDict(get_random_action_id)
        self.probabilities = dict()

    def __setitem__(self, key, value):
        self.lazy_dict[key] = value
        w = get_eps_soft_distribution(value)
        self.probabilities[key] = w

    def __getitem__(self, item):
        a = self.lazy_dict[item]
        if item not in self.probabilities:
            w = get_eps_soft_distribution(a)
            self.probabilities[item] = w
        return np.random.choice(IDs, p=self.probabilities[item])
