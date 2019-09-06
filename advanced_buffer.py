import numpy as np
from topology.state_2_topology import find_intersections
from state_encoder import *
import pdb

def get_reward_key(reward, start_state):
    intersections = find_intersections(start_state)
    num_segs = len(intersections) * 2 + 1
    # using _ and - so that the string is valid tensorflow scope name
    if reward.get('move')=='R1':
        return "move-R1_left-%d_sign-%d" % (reward['left'], reward['sign'])
    if reward.get('move')=='R2' and reward.get('over_idx') == reward.get('under_idx'):
        return "move-R2_left-%d_over_before_under-%d" % (reward['left'], reward['over_before_under'])
    if reward.get('move')=='R2':
        return "move-R2_left-%d_diff" %(reward['left'])
    if reward.get('move')=='cross':
        over = reward.get('over_idx')==0 or reward.get('over_idx')==(num_segs-1)
        if over:
            return "move-cross_endpoint-over_sign-%d" % (reward['sign'])
        else:
            return "move-cross_endpoint-under_sign-%d" % (reward['sign'])
    pdb.set_trace()
    raise NotImplementedError

class Buffer(object):
    # gets obs, actions, rewards, mu's, (states, masks), dones
    def __init__(self, reward_key, size=50000):
        self.size = size
        # Memory
        self.obs = None
        self.actions = None
        self.over_seg_range = None
        self.under_seg_range = None
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
            self.over_seg_range = np.empty([self.size, 2], dtype=np.int32)
            self.under_seg_range = np.empty([self.size, 2], dtype=np.int32)

        if get_reward_key(rewards, obs) == self.reward_key:
            self.obs[self.next_idx] = obs
            self.actions[self.next_idx] = actions
            _, over_range, under_range = state_2_buffer(obs, rewards)
            self.over_seg_range[self.next_idx] = over_range
            self.under_seg_range[self.next_idx] = under_range
            self.next_idx = (self.next_idx + 1) % self.size
            self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def get(self, batch_size):
        # returns
        # obs [batch, 64, 3]
        # actions [batch, 6]
        # rewards [batch, ]
        assert self.can_sample()

        idx = np.random.randint(0, self.num_in_buffer, batch_size)
        obs = self.obs[idx]
        actions = self.actions[idx]
        over_seg_range = self.over_seg_range[idx]
        under_seg_range = self.under_seg_range[idx]

        seg_dicts = [buffer_2_model(ob,over,under) for ob,over,under in
                     zip(obs, over_seg_range, under_seg_range)]
        seg_dicts = zip(*seg_dicts)
        _, over, under = seg_dicts
        over_seg_dict = pad_batch(over)
        under_seg_dict = pad_batch(under)

        obs = np.array(obs)
        actions = np.array(actions)

        return obs, actions, np.ones((batch_size,)), over_seg_dict, under_seg_dict

    def dump(self):
        np.savez(self.reward_key+'_buffer.npz',actions=self.actions[:self.num_in_buffer],
                                               obs=self.obs[:self.num_in_buffer],
                                               over_seg_range=self.over_seg_range[:self.num_in_buffer],
                                               under_seg_range=self.under_seg_range[:self.num_in_buffer],
                                               )

    def load(self, np_file):
        data = np.load(np_file)
        self.num_in_buffer = data['actions'].shape[0]
        self.obs = np.empty([self.size] + list(data['obs'][0].shape), dtype=np.float32)
        self.actions = np.empty([self.size] + list(data['actions'][0].shape), dtype=np.float32)
        self.over_seg_range = np.empty([self.size, 2], dtype=np.int32)
        self.under_seg_range = np.empty([self.size, 2], dtype=np.int32)
        self.obs[:self.num_in_buffer]=data['obs']
        self.actions[:self.num_in_buffer]=data['actions']
        self.over_seg_range[:self.num_in_buffer]=data['over_seg_range']
        self.under_seg_range[:self.num_in_buffer]=data['under_seg_range']
        self.next_idx = self.num_in_buffer
