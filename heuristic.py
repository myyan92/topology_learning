import numpy as np
import pdb
import os, sys, glob
import gin
from knot_env import KnotEnv

def state_ICP_distance(state1, state2):
    # states are 64x2 arrays.
    mean1 = np.mean(state1, axis=0, keepdims=True)
    state1 = state1 - mean1
    mean2 = np.mean(state2, axis=0, keepdims=True)
    state2 = state2 - mean2
    cov = np.dot(state1.transpose(), state2)
    U,S,VT = np.linalg.svd(cov)
    dist = np.sum(state1**2)+np.sum(state2**2) - 2*np.sum(S)
    R = np.dot(U, VT)
    t = mean1 - np.dot(mean2, R.transpose())
    # state1 <- state2.dot(R.T)+t
    return dist, R, t

def interpolate_heuristic(anno_state, anno_action, query_state):
    T = 0.2
    weights = []
    aligned_actions = []
    for st,ac in zip(anno_state, anno_action):
        dist, R, t = state_ICP_distance(query_state, st)
        weights.append(dist)
        action = ac.copy()
        action[1:3] = np.dot(action[np.newaxis, 1:3], R.transpose()) + t
        action[3:5] = np.dot(action[np.newaxis, 3:5], R.transpose()) + t
        aligned_actions.append(action)

    weights = np.array(weights)
    weights = np.exp(-weights / T)
    weights = weights / np.sum(weights)
    return np.sum(weights[:,np.newaxis]*np.array(aligned_actions), axis=0)

def load_heuristic(folder):
    files = glob.glob(os.path.join(folder, '*.txt'))
    root = '/'.join(folder.split('/')[:-1])
    anno_states, anno_actions = [],[]
    for f in files:
        points = np.loadtxt(f)
        states = f.split('/')[-1].split('_')[0]
        states = os.path.join(root, '%s.txt'%(states))
        states = np.loadtxt(states)
        dists = np.linalg.norm(states[:,:2]-points[0], axis=1)
        node = np.argmin(dists)
        action_mean = [node/63.0]+points[1].tolist()+points[2].tolist()+[0.1]
        action_mean = np.array(action_mean)
        anno_states.append(states[:,:2])
        anno_actions.append(action_mean)
    return anno_states, anno_actions

if __name__ == "__main__":
    folder = '2intersect_states/annotate_2on4'
    anno_states, anno_actions = load_heuristic(folder)

    query_state = anno_states[0]
    interpolate_action = interpolate_heuristic(anno_states, anno_actions, query_state)
    action_std = np.array([0.03,0.05,0.05,0.05,0.05,0.05])

    gin_config_file = sys.argv[1]
    gin.parse_config_file(gin_config_file)

    env = KnotEnv(parallel=64)
    eval_size=128
    rewards = []
    for _ in range(eval_size//env.parallel):
        obs = env.reset()
        actions = []
        for ob in obs:
            action = interpolate_heuristic(anno_states, anno_actions, ob[:,:2])
            actions.append(np.random.normal(loc=action, scale=action_std))
        end_obs, r, done, _ = env.step(actions)
        rewards.extend(list(r))
    print(rewards)
    success = [1 if r.get('move', None)=='cross' and r['over_idx']==2 and r['under_idx']==4 and r['sign']==-1 else 0 for r in rewards]
    print(np.mean(success))
