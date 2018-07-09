""" Exercise 5.10: Racetrack - Definitions """

import numpy as np

np.random.seed(seed=1234)

ACTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]  # 9 possible actions
n = len(ACTIONS)
IDs = list(range(0, n))
VELOCITY_COMPONENT_MAX = 5

eps = 0.05
gamma = 1.
noise = 0.1


def get_random_action_id():
    return np.random.randint(0, n)


def get_eps_soft_distribution(a):
    return [eps/n] * a + [1-eps+eps/n] + [eps/n]*(n-a-1)


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

    for action_id in range(0, n):
        if val is None:
            if (state, action_id) in q.data:
                val = q[(state, action_id)]
                idx = action_id
            continue
        if (state, action_id) in q.data:
            # print("action_id: ", action_id, ", value: ", q.data[(state, action_id)])
            if val < q[(state, action_id)]:
                val = q[(state, action_id)]
                # print("v: ", v)
                idx = action_id
    return idx


class SoftPolicy(object):

    def __init__(self):
        self.lazy_dict = LazyDict(get_random_action_id)
        self.probabilities = dict()

    def __setitem__(self, key, value):
        self.lazy_dict[key] = value
        # self.lazy_dict.insert(key, value)
        w = get_eps_soft_distribution(value)
        self.probabilities[key] = w

    def __getitem__(self, item):
        # item here is a state
        a = self.lazy_dict[item]
        # print("a: ", a)
        if item not in self.probabilities:
            w = get_eps_soft_distribution(a)
            self.probabilities[item] = w
        return np.random.choice(IDs, p=self.probabilities[item])


class Track(object):

    def __init__(self, cols, rows, left_boundary, right_boundary):
        self.cols = cols
        self.rows = rows
        self.lb = left_boundary
        self.rb = right_boundary

    def __str__(self):
        """
        Create simple display text
        """

        result = ''
        for i in range(self.rows-1, -1, -1):
            j = 0
            while j < self.lb[i]:
                result += '-'
                j += 1
            result += '1'
            j += 1
            while j < self.rb[i]:
                result += '0'
                j += 1
            if j < self.cols:
                result += '2'
                j += 1
            while j < self.cols:
                result += '-'
                j += 1
            result += '\n'
        return result

    def get_valid_positions(self):
        positions = []
        for i in range(self.rows - 1, -1, -1):
            j = 0
            while j < self.lb[i]:
                j += 1
            j += 1
            while (j < self.rb[i]) and (j != self.cols-1):
                positions.append([j, i])
                j += 1
            if j < self.cols:
                j += 1
            while j < self.cols:
                j += 1
        return positions

    def check_car_trajectory(self, car, projection):
        """
        Check the projected path of car.

        Possible outcomes:
            0: path is within track boundaries
            1: path intersects the track boundary
            2: path intersects the finish line
        """

        x1, y1 = car.pos
        x2, y2 = projection

        if x1 == x2:
            y = y1 + 1
            while y <= y2:
                if x1 <= self.lb[y]:
                    return 1
                y += 1
        elif y1 == y2:
            x = x1 + 1
            while x <= x2:
                if self.rb[y1] <= x:
                    return 1
                if x == self.cols - 1:
                    return 2
                x += 1
        else:
            k = (y1-y2) / (x1-x2)
            q = (y2*x1 - y1*x2) / (x1-x2)
            q = 0.5*(1-k) + q  # translation to the centre of the cells
            y = y1 + 1
            while y <= y2:
                x = (y - q) / k
                if self.rb[y - 1] == self.cols and x > self.cols - 1:
                    return 2
                if self.lb[y] + 1 > x:
                    return 1
                if self.rb[y - 1] < x:
                    return 1
                y += 1

            # checking the end point:
            if self.lb[y2] >= x2 or x2 >= self.rb[y2]:
                return 1
            if x2 >= self.cols-1:
                return 2

        return 0


class FinishException(Exception):
    pass


class Car(object):

    def __init__(self, noise=0, pos=None, v=None):

        self.noise = noise

        if pos is None:
            x = np.random.randint(L[0] + 1, R[0])
            self.pos = [x, 0]
        else:
            self.pos = pos
        if v is None:
            self.v = [0, 0]
        else:
            self.v = v
        # self.history = [(self.x, self.y), ]
        # self.actions = []

    def velocity_update(self, dv):

        if self.noise > np.random.rand():
            # print("noise applied...")
            dv = [0, 0]

        vx_new = self.v[0] + dv[0]
        vy_new = self.v[1] + dv[1]

        if vx_new == 0 and vy_new == 0:
            return

        if (0 <= vx_new < VELOCITY_COMPONENT_MAX) and \
                (0 <= vy_new < VELOCITY_COMPONENT_MAX):
            self.v[0] = vx_new
            self.v[1] = vy_new

    def position_update(self):

        xp = self.pos[0] + self.v[0]
        yp = self.pos[1] + self.v[1]

        projection = (xp, yp)
        status = TRACK.check_car_trajectory(self, projection)

        if status == 1:  # "out":
            self.pos[0] = np.random.randint(L[0] + 1, R[0])
            self.pos[1] = 0
            self.v[0] = 0
            self.v[1] = 0
            # self.history = [(self.x, self.y)]
            # self.actions = []

        elif status == 0:  # "OK":
            self.pos[0] = xp
            self.pos[1] = yp
            # self.history.append((self.x, self.y))
            # self.actions.append((self.vx, self.vy))

        else:
            raise FinishException
            # self.x = xp
            # self.y = yp
            # self.history.append((self.x, self.y))
            # self.actions.append((self.vx, self.vy))
            # self.history.append("FINNISH")


class Episode(object):

    def __init__(self, policy, noise=0, car=None):
        self.policy = policy
        # self.noise = noise
        if car is None:
            self.car = Car()
        else:
            self.car = car
        self.car.noise = noise
        self.state_history = []
        self.action_history = []

    def __iter__(self):
        return self

    def __next__(self):

        current_state = (tuple(self.car.pos), tuple(self.car.v))
        self.state_history.append(current_state)

        dv = ACTIONS[self.policy[current_state]]

        self.action_history.append(ACTIONS.index(dv))
        self.car.velocity_update(dv)

        try:
            self.car.position_update()
        except FinishException:
            # saving the terminal state (although it is not used in the calculations)
            xp = self.car.pos[0] + self.car.v[0]
            yp = self.car.pos[1] + self.car.v[1]
            self.state_history.append(((xp, yp), tuple(self.car.v)))
            raise StopIteration


def generate_random_track(m, n):
    l, r = [], []
    lp = np.random.randint(0, m // 2)
    l.append(lp)
    rp = np.random.randint(lp+2, m)
    r.append(rp)
    for i in range(n-2):
        coin = np.random.rand()
        if coin < 1 - (i / (n+1)) ** 2:
            step = 0
        else:
            step = np.random.choice([1, 2, 3, 4], p=[0.5, 0.2, 0.2, 0.1])
        l.append(l[i] + step if l[i] + step < m else (m-1))
        coin = np.random.rand()
        if coin < 1 - (i / (n+1)) ** 2:
            step = 0
        else:
            step = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
        r.append(r[i] + step if r[i] + step < m else m)
    r.append(m-1)
    l.append(m-1)
    return l, r


M = 15
N = 32
width = 6

# L, R = generate_random_track(M, N)
# L = [0] * (N-1) + [M-1]
# R = [WIDTH+1] * (N - WIDTH-2) + [WIDTH+2] + [M] * WIDTH + [M-1]
L = [0] * (N - 1 - width + 4) + [width] * (width - 4) + [M - 1]
R = [width + 1] * (N - width - 2) + [width + 2] + [M] * width + [M - 1]

print(L)
print(R)

TRACK = Track(M, N, L, R)
print(TRACK)
