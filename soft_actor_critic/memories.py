import random
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler


class ReplayMemory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class PrioritizedMemory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask, loss):
        if len(self.buffer) < self.memory_size:
            self.priorities.append(None)
            self.buffer.append(None)
            self.position = len(self.buffer) - 1
        else:
            self.position = np.argmin(self.priorities)
        self.priorities[self.position] = loss
        self.buffer[self.position] = (state, action, reward, next_state, mask)

    def sample(self, batch_size):
        indices = list(WeightedRandomSampler(make_priority_probabilities(np.array(self.priorities).copy()), batch_size, replacement=False))
        state, action, reward, next_state, done = map(np.stack, zip(*[self.buffer[i] for i in indices]))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def make_priority_probabilities(priorities):
    probabilities = priorities - np.min(priorities)
    return probabilities / np.max(probabilities)
