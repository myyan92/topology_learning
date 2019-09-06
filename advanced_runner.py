import numpy as np
import random
from advanced_buffer import get_reward_key
from model_stats import ModelStats
from planner import get_intended_action, encode
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

        self.gamma = gamma # TODO for RRT env?

    def run(self, sess):
        intended_action = get_intended_action(self.obs[0]) # assumes all the same for each batch
        model = self.model_dict[get_reward_key(intended_action, self.obs[0])] # hard coded
        model_inputs = encode(self.obs, [intended_action]*len(self.obs))
        actions = model.predict_batch(sess, *model_inputs, explore=True)
        actions = np.clip(actions, self.env.action_low, self.env.action_high)
        obs, rewards, dones, infos = self.env.step(actions)
        for ob,ac,r in zip(self.obs, actions, rewards):
            if hash_dict(r) == hash_dict(intended_action):
                reward_key = get_reward_key(r, ob)
                if reward_key in self.buffer_dict:
                    self.buffer_dict[reward_key].put(ob, ac, r)
            stats = self.model_stats_dict[model.scope]
            stat_reward = 1.0 if hash_dict(r) == hash_dict(intended_action) else 0.0
            stats.put(stat_reward)
        self.obs = self.env.reset()
        # TODO how to coordinate multiple policies?
