import numpy as np
from advanced_buffer import get_reward_key
from state_encoder import unifying_transform_encode, state_2_buffer

actions = [
    {'idx':0, 'left':-1, 'move':'R1', 'sign':1},
    {'idx':0, 'left':1, 'move':'R1', 'sign':-1},
    {'idx':1, 'left':-1, 'move':'R1', 'sign':-1},
    {'idx':1, 'left':-1, 'move':'R1', 'sign':1},
    {'idx':1, 'left':1, 'move':'R1', 'sign':-1},
    {'idx':1, 'left':1, 'move':'R1', 'sign':1},
    {'idx':2, 'left':-1, 'move':'R1', 'sign':-1},
    {'idx':2, 'left':1, 'move':'R1', 'sign':1},
    {'left':-1, 'move':'R2', 'over_before_under':-1, 'over_idx':1, 'under_idx':0},
    {'left':-1, 'move':'R2', 'over_before_under':-1, 'over_idx':1, 'under_idx':1},
    {'left':-1, 'move':'R2', 'over_before_under':-1, 'over_idx':2, 'under_idx':0},
    {'left':-1, 'move':'R2', 'over_before_under':-1, 'over_idx':2, 'under_idx':1},
    {'left':-1, 'move':'R2', 'over_before_under':-1, 'over_idx':2, 'under_idx':2},
    {'left':-1, 'move':'R2', 'over_before_under':1, 'over_idx':0, 'under_idx':0},
    {'left':-1, 'move':'R2', 'over_before_under':1, 'over_idx':0, 'under_idx':1},
    {'left':-1, 'move':'R2', 'over_before_under':1, 'over_idx':0, 'under_idx':2},
    {'left':-1, 'move':'R2', 'over_before_under':1, 'over_idx':1, 'under_idx':1},
    {'left':-1, 'move':'R2', 'over_before_under':1, 'over_idx':1, 'under_idx':2},
    {'left':1, 'move':'R2', 'over_before_under':-1, 'over_idx':1, 'under_idx':0},
    {'left':1, 'move':'R2', 'over_before_under':-1, 'over_idx':1, 'under_idx':1},
    {'left':1, 'move':'R2', 'over_before_under':-1, 'over_idx':2, 'under_idx':0},
    {'left':1, 'move':'R2', 'over_before_under':-1, 'over_idx':2, 'under_idx':1},
    {'left':1, 'move':'R2', 'over_before_under':-1, 'over_idx':2, 'under_idx':2},
    {'left':1, 'move':'R2', 'over_before_under':1, 'over_idx':0, 'under_idx':0},
    {'left':1, 'move':'R2', 'over_before_under':1, 'over_idx':0, 'under_idx':1},
    {'left':1, 'move':'R2', 'over_before_under':1, 'over_idx':0, 'under_idx':2},
    {'left':1, 'move':'R2', 'over_before_under':1, 'over_idx':1, 'under_idx':1},
    {'left':1, 'move':'R2', 'over_before_under':1, 'over_idx':1, 'under_idx':2},
    {'move':'cross', 'over_idx':0, 'sign':-1, 'under_idx':1},
    {'move':'cross', 'over_idx':0, 'sign':-1, 'under_idx':2},
    {'move':'cross', 'over_idx':0, 'sign':1, 'under_idx':1},
    {'move':'cross', 'over_idx':0, 'sign':1, 'under_idx':2},
    {'move':'cross', 'over_idx':2, 'sign':-1, 'under_idx':0},
    {'move':'cross', 'over_idx':2, 'sign':-1, 'under_idx':1},
    {'move':'cross', 'over_idx':2, 'sign':1, 'under_idx':0},
    {'move':'cross', 'over_idx':2, 'sign':1, 'under_idx':1}
]

def hash_dict(abstract_action):
    tokens = [k+'-'+str(v) for k,v in abstract_action.items()]
    return '_'.join(sorted(tokens))

merged_buffers = {}
for Tac in actions:
    buffer_name = hash_dict(Tac) + '_buffer.npz'
    data = np.load(buffer_name)
    obs = data['obs']
    actions = data['actions']
    obs_u, actions_u = [], []
    over_seg_range_u, under_seg_range_u = [], []
    for ob, ac in zip(obs, actions):
        ob_u, ac_u, Tac_u, transform = unifying_transform_encode(ob, ac, Tac)
        ob_u, over_seg_u, under_seg_u = state_2_buffer(ob_u, Tac_u)
        obs_u.append(ob_u)
        actions_u.append(ac_u)
        over_seg_range_u.append(over_seg_u)
        under_seg_range_u.append(under_seg_u)
    reward_key = get_reward_key(Tac_u, obs_u[0])
    if reward_key in merged_buffers:
        merged_buffers[reward_key]['obs'].extend(obs_u)
        merged_buffers[reward_key]['actions'].extend(actions_u)
        merged_buffers[reward_key]['over_seg_range'].extend(over_seg_range_u)
        merged_buffers[reward_key]['under_seg_range'].extend(under_seg_range_u)
    else:
        merged_buffers[reward_key] = {
                                      'obs':obs_u, 'actions':actions_u,
                                      'over_seg_range':over_seg_range_u,
                                      'under_seg_range':under_seg_range_u
                                     }

for key, buffer in merged_buffers.items():
    save_name = key + '_init_buffer.npz'
    np.savez(save_name, **buffer)
