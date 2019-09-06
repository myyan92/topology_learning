import numpy as np
import random
from advanced_buffer import Buffer, get_reward_key
from knot_env import KnotEnv
import pdb

def hash_dict(abstract_action):
    tokens = [k+':'+str(v) for k,v in abstract_action.items()]
    return ' '.join(sorted(tokens))

env = KnotEnv(60)
buffer = Buffer('move-cross_endpoint-over_sign-1', size=1000)

while buffer.num_in_buffer < 100:
    obs = env.reset()
    intended_action = {'move':'cross', 'over_idx':2, 'under_idx':0, 'sign':1}
    action_gaussian_mean = np.array([0.95, 0.18, 0.1, 0.15, -0.1, 0.05])
    action_gaussian_std = np.array([0.03, 0.05, 0.05, 0.05, 0.05, 0.03])
#    intended_action = {'move':'cross', 'over_idx':2, 'under_idx':1, 'sign':1}
#    action_gaussian_mean = np.array([0.95, 0.30, 0.1, 0.32, -0.1, 0.05])
#    action_gaussian_std = np.array([0.03, 0.05, 0.05, 0.05, 0.05, 0.03])
    actions = np.random.normal(loc=action_gaussian_mean, scale=action_gaussian_std, size=(env.parallel, 6))
    actions = np.clip(actions, env.action_low, env.action_high)
    _, rewards, dones, infos = env.step(actions)
    for ob,ac,r in zip(obs, actions, rewards):
        if hash_dict(r) == hash_dict(intended_action):
            buffer.put(ob, ac, r)
    print(buffer.num_in_buffer)
    #env.render()

buffer.dump()
