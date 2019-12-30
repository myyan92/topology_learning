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

root = './1loop_states/'
#folders = ['annotate_0on1', 'annotate_0on2', 'annotate_2on1', 'annotate_2on0']
folders = ['annotate_2on0']
reward_key = 'move-cross_endpoint-over_sign-1'
buffer = Buffer(reward_key, size=3000)

idx=0
for folder in folders:
    files = glob.glob(os.path.join(root, folder, '*.txt'))
    for f in files:
        idx += 1
        points = np.loadtxt(f)
        states = f.split('/')[-1].split('_')[0]
        states = os.path.join(root, '%s.txt'%(states))
        states = np.loadtxt(states)
        dists = np.linalg.norm(states[:,:2]-points[0], axis=1)
        node = np.argmin(dists)
        action_mean = [node/63.0]+points[1].tolist()+points[2].tolist()+[0.05]
        action_mean = np.array(action_mean)
        action_std = np.array([0.03,0.05,0.05,0.05,0.05,0.05])
        while buffer.num_in_buffer < 100*idx:
            env.start_state = [states]*env.parallel
            obs = env.start_state
            actions = np.random.normal(loc=action_mean, scale=action_std,
                                       size=(env.parallel, 6))
            actions = np.clip(actions, env.action_low, env.action_high)
            end_obs, rewards, dones, infos = env.step(actions)
            for ob,ac,r in zip(obs, actions, rewards):
                if r.get('move',None)=='cross' and r.get('over_idx',None) in [0,2]:
                    if r['over_idx']==2 and r['sign']==-1:
                        ob, ac, r, _ = unifying_transform_encode(ob, ac, r)
                    if r['sign'] == 1:
                        buffer.put(ob, ac, 1.0, r)
            print(buffer.num_in_buffer)

buffer.dump()
os.rename(reward_key+'_buffer.npz', reward_key+'_init_buffer.npz')
