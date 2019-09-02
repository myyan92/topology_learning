import numpy as np
import pdb

def get_reward_key(reward):
    #TODO: implement complete method
    # using _ and - so that the string is valid tensorflow scope name
    if reward.get('move')=='R1' and reward.get('idx')==0:
        return "move-R1_left-%d_sign-%d" % (reward['left'], reward['sign'])
    if reward.get('move')=='R2' and reward.get('over_idx')==0 and reward.get('under_idx')==0:
        return "move-R2_left-%d_over_before_under-%d" % (reward['left'], reward['over_before_under'])

class Buffer(object):
    # gets obs, actions, rewards, mu's, (states, masks), dones
    def __init__(self, reward_key, size=50000):
        self.size = size
        # Memory
        self.obs = None
        self.actions = None
        self.reward_key = reward_key

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

    def has_atleast(self, frames):
        return self.num_in_buffer >= frames

    def can_sample(self):
        return self.num_in_buffer > 0

    def put(self, obs, actions, rewards):
        # obs: np.array((64,3))
        # actions: np.array((6,))
        # reward: dict
        if self.obs is None:
            self.obs = np.empty([self.size] + list(obs.shape), dtype=np.float32)
            self.actions = np.empty([self.size] + list(actions.shape), dtype=np.float32)

        if get_reward_key(rewards) == self.reward_key:
            self.obs[self.next_idx] = obs
            self.actions[self.next_idx] = actions
            self.next_idx = (self.next_idx + 1) % self.size
            self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def take(self, x, idx):
        out = np.empty([len(idx)] + list(x.shape[1:]), dtype=x.dtype)
        for i in range(len(idx)):
            out[i] = x[idx[i]]
        return out

    def get(self, batch_size):
        # returns
        # obs [batch, 64, 3]
        # actions [batch, 6]
        # rewards [batch, ]
        assert self.can_sample()

        idx = np.random.randint(0, self.num_in_buffer, batch_size)
        take = lambda x: self.take(x, idx)
        obs = take(self.obs)
        actions = take(self.actions)
        return obs, actions, np.ones((batch_size,))

    def __del__(self):
        np.savez(self.reward_key+'_buffer.npz',actions=self.actions[:self.num_in_buffer],
                                               obs=self.obs[:self.num_in_buffer])

    def load(self, np_file):
        data = np.load(np_file)
        self.num_in_buffer = data['actions'].shape[0]
        self.obs = np.empty([self.size] + list(data['obs'][0].shape), dtype=np.float32)
        self.actions = np.empty([self.size] + list(data['actions'][0].shape), dtype=np.float32)
        self.obs[:self.num_in_buffer]=data['obs']
        self.actions[:self.num_in_buffer]=data['actions']
        self.next_idx = self.num_in_buffer
