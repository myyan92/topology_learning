import numpy as np
import random
from advanced_buffer import get_reward_key
from model_stats import ModelStats
from planner import get_intended_action, encode
from topology.state_2_topology import state2topology
from state_encoder import unifying_transform_encode, unifying_transform_decode
import pdb

def hash_dict(abstract_action):
    tokens = [k+':'+str(v) for k,v in abstract_action.items()]
    return ' '.join(sorted(tokens))


class Runner(object):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, models, model_stats, buffers, gamma=0.99):
        self.env = env
        self.model_dict = {model.scope:model for model in models}
        self.model_stats_dict = {model_stat.model_name:model_stat for model_stat in model_stats}
        self.buffer_dict = {buffer.reward_key:buffer for buffer in buffers}
        self.obs = env.reset()

        self.count = 1000 # temp
        self.gamma = gamma # TODO for RRT env?

    def run(self, sess):
        intended_action = {'move':'cross', 'over_idx':0, 'under_idx':1, 'sign':-1}
        trans_obs = []
        for obs in self.obs:
            obs_u, _, trans_intended_action, transform = unifying_transform_encode(obs, None, intended_action)
            trans_obs.append(obs_u)
        model = self.model_dict[get_reward_key(trans_intended_action, trans_obs[0])] # hard coded
        model_inputs = encode(trans_obs, [trans_intended_action]*len(self.obs))
#        trans_actions = model.predict_batch(sess, trans_obs, explore=True)
        trans_actions = model.predict_batch(sess, *model_inputs, explore=False)
        trans_actions = np.clip(trans_actions, self.env.action_low, self.env.action_high)
        actions = []
        for tac in trans_actions:
            _, ac, _ = unifying_transform_decode(None, tac, None, transform)
            actions.append(ac)
        actions = np.array(actions)
        obs, rewards, dones, infos = self.env.step(actions)
        batch_r = []
        for ob,ac,r in zip(self.obs, actions, rewards):
            if hash_dict(r) == hash_dict(intended_action):
                reward_key = get_reward_key(r, ob)
                if reward_key in self.buffer_dict:
                    self.buffer_dict[reward_key].put(ob, ac, r)
            stats = self.model_stats_dict[model.scope]
            stat_reward = 1.0 if hash_dict(r) == hash_dict(intended_action) else 0.0
            stats.put(stat_reward)
            batch_r.append(stat_reward)

#        for end_state, r in zip(obs, batch_r):
#            if r == 1:
#                np.savetxt('%03d.txt'%(self.count), end_state)
#                self.count += 1

#        start_topo, _ = state2topology(self.obs[0], update_edges=True, update_faces=False)
#        start_sign = start_topo.points[1].sign
#        print(hash_dict(intended_action), ' start sign %d :'%(start_sign), np.mean(batch_r))
        self.env.render()
        self.obs = self.env.reset()
        # TODO how to coordinate multiple policies?
