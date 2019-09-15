# only a hacking solution right now.
import random
import pdb
from state_encoder import *
from topology.representation import AbstractState
from topology.state_2_topology import state2topology
def get_intended_action(obs):
    # obs must have a topological state trivial state->R1,left-1, sign-1.
    possible_actions = [
                       {'move':'cross', 'over_idx':0, 'under_idx':1, 'sign':1},
                       {'move':'cross', 'over_idx':0, 'under_idx':1, 'sign':-1},
                       {'move':'cross', 'over_idx':0, 'under_idx':2, 'sign':1},
                       {'move':'cross', 'over_idx':0, 'under_idx':2, 'sign':-1},
                       {'move':'cross', 'over_idx':2, 'under_idx':0, 'sign':1},
                       {'move':'cross', 'over_idx':2, 'under_idx':0, 'sign':-1},
                       {'move':'cross', 'over_idx':2, 'under_idx':1, 'sign':1},
                       {'move':'cross', 'over_idx':2, 'under_idx':1, 'sign':-1},
                       {'move':'R1', 'idx':0, 'left':1, 'sign':1},
                       {'move':'R1', 'idx':0, 'left':1, 'sign':-1},
                       {'move':'R1', 'idx':0, 'left':-1, 'sign':1},
                       {'move':'R1', 'idx':0, 'left':-1, 'sign':-1},
                       {'move':'R1', 'idx':1, 'left':1, 'sign':1},
                       {'move':'R1', 'idx':1, 'left':1, 'sign':-1},
                       {'move':'R1', 'idx':1, 'left':-1, 'sign':1},
                       {'move':'R1', 'idx':1, 'left':-1, 'sign':-1},
                       {'move':'R1', 'idx':2, 'left':1, 'sign':1},
                       {'move':'R1', 'idx':2, 'left':1, 'sign':-1},
                       {'move':'R1', 'idx':2, 'left':-1, 'sign':1},
                       {'move':'R1', 'idx':2, 'left':-1, 'sign':-1},
                       {'move':'R2', 'over_idx':0, 'under_idx':0, 'left':1, 'over_before_under':1},
                       {'move':'R2', 'over_idx':0, 'under_idx':0, 'left':-1, 'over_before_under':1},
                       {'move':'R2', 'over_idx':1, 'under_idx':1, 'left':1, 'over_before_under':1},
                       {'move':'R2', 'over_idx':1, 'under_idx':1, 'left':-1, 'over_before_under':1},
                       {'move':'R2', 'over_idx':1, 'under_idx':1, 'left':1, 'over_before_under':-1},
                       {'move':'R2', 'over_idx':1, 'under_idx':1, 'left':-1, 'over_before_under':-1},
                       {'move':'R2', 'over_idx':2, 'under_idx':2, 'left':1, 'over_before_under':-1},
                       {'move':'R2', 'over_idx':2, 'under_idx':2, 'left':-1, 'over_before_under':-1},
                       {'move':'R2', 'over_idx':0, 'under_idx':1, 'left':1, 'over_before_under':1},
                       {'move':'R2', 'over_idx':0, 'under_idx':1, 'left':-1, 'over_before_under':1},
                       {'move':'R2', 'over_idx':0, 'under_idx':2, 'left':1, 'over_before_under':1},
                       {'move':'R2', 'over_idx':0, 'under_idx':2, 'left':-1, 'over_before_under':1},
                       {'move':'R2', 'over_idx':2, 'under_idx':0, 'left':1, 'over_before_under':-1},
                       {'move':'R2', 'over_idx':2, 'under_idx':0, 'left':-1, 'over_before_under':-1},
                       {'move':'R2', 'over_idx':2, 'under_idx':1, 'left':1, 'over_before_under':-1},
                       {'move':'R2', 'over_idx':2, 'under_idx':1, 'left':-1, 'over_before_under':-1}
                       ]
    success = False
    while not success:
        topo_state, _ = state2topology(obs, update_edges=True, update_faces=True)
        intended_action = random.choice(possible_actions)
        move = intended_action.pop('move')
        try:
            if move=='cross':
                success = topo_state.cross(**intended_action)
            elif move=='R1':
                success = topo_state.Reide1(**intended_action)
            elif move=='R2':
                success = topo_state.Reide2(**intended_action)
        except:
            pdb.set_trace()
        intended_action['move']=move
    return intended_action

def encode(obs, intended_actions):
    encode_dicts = [buffer_2_model(*state_2_buffer(ob, ac)) for ob,ac in
                    zip(obs, intended_actions)]
    encode_dicts = zip(*encode_dicts)
    obs, over, under = encode_dicts
    obs = np.array(obs)
    over_seg_dict = pad_batch(over)
    under_seg_dict = pad_batch(under)
    return obs, over_seg_dict, under_seg_dict
