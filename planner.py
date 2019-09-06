# only a hacking solution right now.
import random
from state_encoder import *

def get_intended_action(obs):
    # obs must have a topological state trivial state->R1,left-1, sign-1.
    if random.random() > 0.5:
        return {'move':'cross', 'over_idx':2, 'under_idx':0, 'sign':1}
    else:
        return {'move':'cross', 'over_idx':2, 'under_idx':1, 'sign':1}

def encode(obs, intended_actions):
    encode_dicts = [buffer_2_model(*state_2_buffer(ob, ac)) for ob,ac in
                    zip(obs, intended_actions)]
    encode_dicts = zip(*encode_dicts)
    obs, over, under = encode_dicts
    obs = np.array(obs)
    over_seg_dict = pad_batch(over)
    under_seg_dict = pad_batch(under)
    return obs, over_seg_dict, under_seg_dict
