import numpy as np
from topology.state_2_topology import find_intersections
from topology.representation import AbstractState
from state_encoder import *
import pdb

def get_reward_key(reward, start_state):
    if isinstance(start_state, np.ndarray):
        intersections = find_intersections(start_state)
        num_segs = len(intersections) * 2 + 1
    elif isinstance(start_state, AbstractState):
        num_segs = start_state.pts+1
    else:
        raise NotImplementedError
    # using _ and - so that the string is valid tensorflow scope name
    if reward.get('move') is None:
        return None
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
    def __init__(self, reward_key, size=50000, filter_success=True):
        self.size = size
        self.filter_success = filter_success
        # Memory
        self.obs = None
        self.actions = None
        self.rewards = None
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

    def put(self, obs, actions, rewards, intended_action):
        # obs: np.array((64,3))
        # actions: np.array((6,))
        # reward: dict
        if self.obs is None:
            self.obs = np.empty([self.size] + list(obs.shape), dtype=np.float32)
            self.actions = np.empty([self.size] + list(actions.shape), dtype=np.float32)
            self.over_seg_range = np.empty([self.size, 2], dtype=np.int32)
            self.under_seg_range = np.empty([self.size, 2], dtype=np.int32)
            if not self.filter_success:
                self.rewards = np.empty([self.size], dtype=np.float32)

        if (not self.filter_success) or rewards>0.0:
            self.obs[self.next_idx] = obs
            self.actions[self.next_idx] = actions
            _, over_range, under_range = state_2_buffer(obs, intended_action)
            self.over_seg_range[self.next_idx] = over_range
            self.under_seg_range[self.next_idx] = under_range
            if not self.filter_success:
                self.rewards[self.next_idx] = rewards
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

        if not self.filter_success:
            rewards = np.array(self.rewards[idx])
        else:
            rewards = np.ones((batch_size,))
        return obs, actions, rewards, over_seg_dict, under_seg_dict

    def augment(self, obs, actions, over_seg_dict, under_seg_dict):
        rotations = np.random.uniform(0,np.pi*2, size=(obs.shape[0],))
        translations = np.random.uniform(-0.1,0.1,size=(obs.shape[0],1,2))
        rotations = np.array([[np.cos(rotations), np.sin(rotations)],
                              [-np.sin(rotations), np.cos(rotations)]]).transpose((2,0,1))
        obs = obs.copy()
        actions = actions.copy()
        obs[:,:,:2] = np.matmul(obs[:,:,:2], rotations) + translations
        actions[:,1:3] = np.matmul(actions[:,np.newaxis,1:3], rotations)[:,0,:] + translations[:,0,:]
        actions[:,3:5] = np.matmul(actions[:,np.newaxis,3:5], rotations)[:,0,:] + translations[:,0,:]
        over_seg_obs = over_seg_dict['obs'].copy()
        over_seg_obs[:,:,:2] = np.matmul(over_seg_obs[:,:,:2], rotations) + translations
        for i,l in enumerate(over_seg_dict['length']):
            over_seg_obs[i,l:] = 0.0
        under_seg_obs = under_seg_dict['obs'].copy()
        under_seg_obs[:,:,:2] = np.matmul(under_seg_obs[:,:,:2], rotations) + translations
        for i,l in enumerate(under_seg_dict['length']):
            under_seg_obs[i,l:] = 0.0
        over_seg_dict = {'obs':over_seg_obs, 'pos':over_seg_dict['pos'], 'length':over_seg_dict['length']}
        under_seg_dict = {'obs':under_seg_obs, 'pos':under_seg_dict['pos'], 'length':under_seg_dict['length']}
        return obs, actions, over_seg_dict, under_seg_dict


    def dump(self):
        if not self.filter_success:
            rewards = self.rewards[:self.num_in_buffer]
        else:
            rewards = np.ones((self.num_in_buffer,))
        np.savez(self.reward_key+'_buffer.npz',actions=self.actions[:self.num_in_buffer],
                                               obs=self.obs[:self.num_in_buffer],
                                               rewards=rewards,
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
        if not self.filter_success:
            self.rewards = np.empty([self.size], dtype=np.float32)
            self.rewards[:self.num_in_buffer]=data['rewards']
        self.next_idx = self.num_in_buffer

    def append(self, np_file):
        if self.obs is None:
            self.load(np_file)
            return
        data = np.load(np_file)
        length = data['actions'].shape[0]
        length = min(length, self.size-self.num_in_buffer)
        self.obs[self.num_in_buffer:self.num_in_buffer+length]=data['obs'][:length]
        self.actions[self.num_in_buffer:self.num_in_buffer+length]=data['actions'][:length]
        self.over_seg_range[self.num_in_buffer:self.num_in_buffer+length]=data['over_seg_range'][:length]
        self.under_seg_range[self.num_in_buffer:self.num_in_buffer+length]=data['under_seg_range'][:length]
        if not self.filter_success:
            self.rewards[self.num_in_buffer:self.num_in_buffer+length]=data['rewards'][:length]
        self.num_in_buffer += length
        self.next_idx += length

