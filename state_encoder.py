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


def mirror(ob, action, reward):
    # flip the y axis

    ob=ob.copy() if ob is not None else None
    action=action.copy() if action is not None else None
    reward=reward.copy() if reward is not None else None

    if ob is not None:
        ob[:,1]=-ob[:,1]
    if action is not None:
        action[2]=-action[2]
        action[4]=-action[4]
    if reward is not None:
        if 'left' in reward:
            reward['left']=-reward['left']
        if 'sign' in reward:
            reward['sign']=-reward['sign']
    return ob, action, reward

def reverse(ob, action, reward):
    # flip the order of points in the ob

    ob=ob.copy() if ob is not None else None
    action=action.copy() if action is not None else None
    reward=reward.copy() if reward is not None else None

    if ob is not None:
        ob=ob[::-1]
    if action is not None:
        action[0]=1-action[0]
    if reward is not None:
        assert(ob is not None)
        if 'left' in reward:
            reward['left']=-reward['left']
        if 'over_before_under' in reward:
            reward['over_before_under']=-reward['over_before_under']
        intersections = find_intersections(ob)
        num_segs = len(intersections) * 2 + 1
        for key in ['idx' ,'over_idx', 'under_idx']:
            if key in reward:
                reward[key] = num_segs - reward[key] - 1
    return ob, action, reward


def unifying_transform_encode(ob, action, reward):
    # use mirror and/or reverse so that reward['left']=1, reward['sign']=1,
    # reward['over_before_under']=1
    # return the list of used transforms for decoding.

    """ Usage:
        ob = env.reset()
        intended_action = planner.get_intended_action(ob)
        ob_u, _, intended_action_u, transform = unifying_transform_encode(ob, action=None, intended_action)
        model = model_dict[get_reward_key(intended_action_u)]
        action_u = model.predict(ob_u)
        action = unifying_transform_decode(ob=None, action=action_u, reward=None, transform=transform)
        ob_next, reward, _, _ = env.step(action)
        ob_u, action_u, reward_u, _ = unifying_transform_encode(ob, action, reward)
        # push to buffer
    """

    if reward.get('move') is None:
        return ob, action, reward, []
    transform = []
    if reward.get('move') in ['cross', 'R1'] and reward['sign']==-1:
        ob, action, reward = mirror(ob, action, reward)
        transform.append('mirror')
    if reward.get('move') == 'R1' and reward['left']==-1:
        ob, action, reward = reverse(ob, action, reward)
        transform.append('reverse')
    if reward.get('move') == 'R2' and reward['over_before_under']==-1:
        ob, action, reward = reverse(ob, action, reward)
        transform.append('reverse')
    if reward.get('move') == 'R2' and reward['left']==-1:
        ob, action, reward = mirror(ob, action, reward)
        transform.append('mirror')
    return ob, action, reward, transform

def unifying_transform_decode(ob, action, reward, transform):

    if 'mirror' in transform:
        ob, action, reward = mirror(ob, action, reward)
    if 'reverse' in transform:
        ob, action, reward = reverse(ob, action, reward)
    return ob, action, reward

