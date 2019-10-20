import random
import pdb
from state_encoder import *
from topology.state_2_topology import state2topology
from topology.BFS import generate_next
from advanced_buffer import get_reward_key
import gin

@gin.configurable
def get_random_action(obs, reward_key_list):
    #TODO add a limiting action list?
    topo_state, _ = state2topology(obs, update_edges=True, update_faces=True)
    possible_actions = [ta[1] for ta in generate_next(topo_state)]
    while True:
        intended_action = random.choice(possible_actions)
        unified_action = intended_action.copy()
        if 'left' in unified_action:
            unified_action['left']=1
        if 'sign' in unified_action:
            unified_action['sign']=1
        if get_reward_key(unified_action, obs) in reward_key_list:
            return intended_action

@gin.configurable
def get_fixed_action(obs, reward_key_list, action):
    return action

def encode(obs, intended_actions):
    encode_dicts = [buffer_2_model(*state_2_buffer(ob, ac)) for ob,ac in
                    zip(obs, intended_actions)]
    encode_dicts = zip(*encode_dicts)
    obs, over, under = encode_dicts
    obs = np.array(obs)
    over_seg_dict = pad_batch(over)
    under_seg_dict = pad_batch(under)
    return obs, over_seg_dict, under_seg_dict
