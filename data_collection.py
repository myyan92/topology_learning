import numpy as np
import random
import json
import os
from advanced_buffer import Buffer, get_reward_key
from knot_env import KnotEnv
import pdb

def hash_dict(abstract_action):
    tokens = [k+'-'+str(v) for k,v in abstract_action.items()]
    return '_'.join(sorted(tokens))

env = KnotEnv(60)
env.reset()
with open('data_collection_heuristic.json', 'r') as f:
    annotations = json.load(f)

for annotation_item in annotations[14:16]:
    action_name = hash_dict(annotation_item['intended_action'])
    action_gaussian_mean = np.array(annotation_item['action_gaussian_mean'])
    action_gaussian_std = np.array(annotation_item['action_gaussian_std'])

    print("collecting ", action_name)
    reward_key = get_reward_key(annotation_item['intended_action'], env.start_state[0])
    buffer = Buffer(reward_key, size=1000)

    while buffer.num_in_buffer < 100:
        obs = env.reset()
        actions = np.random.normal(loc=action_gaussian_mean, scale=action_gaussian_std, size=(env.parallel, 6))
        actions = np.clip(actions, env.action_low, env.action_high)
        _, rewards, dones, infos = env.step(actions)
        for ob,ac,r in zip(obs, actions, rewards):
            if hash_dict(r) == action_name:
                buffer.put(ob, ac, r)
        print(buffer.num_in_buffer)
        env.render()

    buffer.dump()
    os.rename(reward_key+'_buffer.npz', action_name+'_init_buffer.npz')
