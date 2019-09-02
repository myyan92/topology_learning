import numpy as np
import pdb

class ModelStats(object):
    def __init__(self, model_name, size=100):
        self.size = size
        self.model_name = model_name
        self.reward = np.zeros([self.size], dtype=np.float32)

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

    def has_atleast(self, frames):
        return self.num_in_buffer >= frames

    def put(self, reward):
        self.reward[self.next_idx] = reward
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def stat(self):
        stat_string = "%s: succeed %d times in the past %d trial" % (self.model_name, np.sum(self.reward), self.num_in_buffer)
        return stat_string
