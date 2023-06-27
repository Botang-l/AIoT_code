from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        rb = batch_size // 100
        data = random.sample(self.memory, batch_size - rb)
        # print(len(data))
        # print(type(data))
        if rb:
            data += list(self.memory)[-rb :]
        # print(len(data))
        # print(type(data))

        return data

    def __len__(self):
        return len(self.memory)


class LSTMdata(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
