import numpy as np
from topology.state_2_topology import find_intersections
import pdb

def state_2_buffer(ob, reward):
    # ob: np.array((64,3))
    #reward: dict

    intersections = find_intersections(ob)
    num_segs = len(intersections) * 2 + 1
    intersections = [i[0] for i in intersections] + [i[1] for i in intersections]
    intersections.sort()
    intersections = [0] + intersections + [64]
    over_idx = reward.get('over_idx', reward.get('idx'))
    over_seg_range = np.array(intersections[over_idx:over_idx+2])
    under_idx =reward.get('under_idx', reward.get('idx'))
    under_seg_range = np.array(intersections[under_idx:under_idx+2])
    return ob, over_seg_range, under_seg_range

def buffer_2_model(ob, over_seg_range, under_seg_range):

    r = over_seg_range
    over_seg_obs = ob[r[0]:r[1]]
    over_seg_pos = np.arange(r[0],r[1])/63.0
    over_seg_length = r[1]-r[0]
    r = under_seg_range
    under_seg_obs = ob[r[0]:r[1]]
    under_seg_pos = np.arange(r[0],r[1])/63.0
    under_seg_length = r[1]-r[0]
    over_seg_dict = {'obs':over_seg_obs, 'pos':over_seg_pos, 'length':over_seg_length}
    under_seg_dict = {'obs':under_seg_obs, 'pos':under_seg_pos, 'length':under_seg_length}
    return ob, over_seg_dict, under_seg_dict

def pad_batch(seg_dicts):

    seg_obs = [dict['obs'] for dict in seg_dicts]
    seg_pos = [dict['pos'] for dict in seg_dicts]
    seg_length = [dict['length'] for dict in seg_dicts]

    max_length = np.amax(seg_length)
    seg_obs = [np.pad(ob,((0,max_length-l),(0,0)), mode='constant')
               for ob,l in zip(seg_obs, seg_length)]
    seg_pos = [np.pad(pos,((0,max_length-l),), mode='constant')
               for pos,l in zip(seg_pos, seg_length)]
    seg_obs = np.array(seg_obs)
    seg_pos = np.array(seg_pos)[:,:,np.newaxis]
    seg_length = np.array(seg_length)

    return  {'obs': seg_obs, 'pos': seg_pos, 'length': seg_length}
