import numpy as np
import random
from buffer import get_reward_key
from model_stats import ModelStats
import pdb

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
        model = random.choice(list(self.model_dict.values()))
        actions = model.predict_batch(sess, self.obs, explore=True)
        actions = np.clip(actions, self.env.action_low, self.env.action_high)
        obs, rewards, dones, infos = self.env.step(actions)
        for ob,ac,r in zip(self.obs, actions, rewards):
            reward_key = get_reward_key(r)
            if reward_key is not None and reward_key in self.buffer_dict:
                self.buffer_dict[reward_key].put(ob, ac, r)
            stats = self.model_stats_dict[model.scope]
            stat_reward = 1.0 if reward_key == model.scope else 0.0
            stats.put(stat_reward)
#        self.env.render() # tmp, for eval
        self.obs = self.env.reset()
        # TODO how to coordinate multiple policies?
