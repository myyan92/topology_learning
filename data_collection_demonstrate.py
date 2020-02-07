import numpy as np
import random
import json
import os
from state_encoder import unifying_transform_encode
from advanced_buffer import Buffer, get_reward_key
import glob
from knot_env import KnotEnv
import pdb

def hash_dict(abstract_action):
    tokens = [k+'-'+str(v) for k,v in abstract_action.items()]
    return '_'.join(sorted(tokens))

env = KnotEnv(64)

root = './1loop_states_new/'
folders = ['annotate_0on1', 'annotate_0on2', 'annotate_2on1', 'annotate_2on0']
reward_key = 'move-cross_endpoint-over_sign-1'
#root = './2intersect_states_new/'
#folders = ['annotate_2on0', 'annotate_2on4']
#reward_key = 'move-cross_endpoint-under_sign-1'

buffer = Buffer(reward_key, size=12000, filter_success=False)

idx=0
for folder in folders:
    files = glob.glob(os.path.join(root, folder, '*.txt'))
    for f in files:
        idx += 1
        points = np.loadtxt(f)
        states = f.split('/')[-1].split('_')[0]
        states = os.path.join(root, '%s.txt'%(states))
        states = np.loadtxt(states)
        obs = 0.5*(states[:64]+states[64:])
        dists = np.linalg.norm(obs[:,:2]-points[0], axis=1)
        node = np.argmin(dists)
        action_mean = [node/63.0]+points[1].tolist()+points[2].tolist()+[0.05]
        action_mean = np.array(action_mean)
        action_std = np.array([0.03,0.05,0.05,0.05,0.05,0.05]) # for 1to2
        action_std = np.array([0.03,0.02,0.02,0.02,0.02,0.05]) # for 2to3

        for _ in range(320//env.parallel):
            env.start_state = [states]*env.parallel
            env.start_obs = [0.5*(st[:64]+st[64:]) for st in states]
            obs = env.start_obs
            actions = np.random.normal(loc=action_mean, scale=action_std,
                                       size=(env.parallel, 6))
            actions = np.clip(actions, env.action_low, env.action_high)
            end_obs, rewards, dones, infos = env.step(actions)

            intended_action = {'move':'cross', 'sign':1}
            intended_action['over_idx'] = int(folder[-4])
            intended_action['under_idx']= int(folder[-1])
            for ob,ac,r in zip(obs, actions, rewards):
                ob, ac, r, _ = unifying_transform_encode(ob, ac, r)
                if hash_dict(r) == hash_dict(intended_action):
                    buffer.put(ob, ac, 1.0, intended_action)
                else:
                    buffer.put(ob, ac, 0.0, intended_action)
            print(buffer.num_in_buffer)

buffer.dump()
os.rename(reward_key+'_buffer.npz', reward_key+'_critic_buffer.npz')
