import numpy as np
import random
from advanced_buffer import get_reward_key
from model_stats import ModelStats
from planner import get_random_action, get_fixed_action, encode
from topology.state_2_topology import state2topology
from state_encoder import unifying_transform_encode, unifying_transform_decode
import gin
import pdb

def hash_dict(abstract_action):
    tokens = [k+':'+str(v) for k,v in abstract_action.items()]
    return ' '.join(sorted(tokens))

@gin.configurable
class Runner(object):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, actor_models, critic_models, model_stats, buffers,
                 topo_action_func=get_random_action, explore=True,
                 eval_save=False, eval_render=False,  gamma=0.99):
        self.env = env
        self.actor_model_dict = {model.scope.replace('_actor',''):model for model in actor_models}
        self.critic_model_dict = {model.scope.replace('_critic',''):model for model in critic_models}
        self.model_stats_dict = {model_stat.model_name:model_stat for model_stat in model_stats}
        self.buffer_dict = {buffer.reward_key:buffer for buffer in buffers}
        self.obs = env.reset()
        self.topo_action_func = topo_action_func
        self.explore = explore
        self.eval_save = eval_save
        self.eval_render = eval_render

        if self.eval_save:
            self.count = 0

        self.gamma = gamma # TODO for RRT env?

    def run(self, sess):
        # HACK
#        for i in range(len(self.obs)):
#            self.env.start_state[i]=np.loadtxt('./1loop_states_new/%03d.txt'%(i))
#        self.env.start_obs=[0.5*(st[:64]+st[64:]) for st in self.env.start_state]
#        self.obs = self.env.start_obs

        model_keys = self.actor_model_dict.keys() if not self.actor_model_dict else self.critic_model_dict.keys()
        intended_actions = [self.topo_action_func(ob, model_keys) for ob in self.obs]
        trans_obs, trans_intended_actions, transforms = [], [], []
        for obs, ia in zip(self.obs, intended_actions):
            obs_u, _, ia_u, transform = unifying_transform_encode(obs, None, ia)
            trans_obs.append(obs_u)
            trans_intended_actions.append(ia_u)
            transforms.append(transform)
        reward_keys = [get_reward_key(ia_u, obs_u) for ia_u, obs_u in zip(trans_intended_actions, trans_obs)]
        trans_actions = [None]*len(self.obs)
        model_keys = set(reward_keys)
        for key in model_keys:
            sublist_trans_obs = [ob_u for ob_u, k in zip(trans_obs, reward_keys) if k==key]
            sublist_trans_ia = [ia_u for ia_u, k in zip(trans_intended_actions, reward_keys) if k==key]
            model_inputs = encode(sublist_trans_obs, sublist_trans_ia)
            actor = self.actor_model_dict.get(key, None)
            critic = self.critic_model_dict.get(key, None)
            if critic is None:
                sublist_trans_actions = actor.predict_batch(sess, *model_inputs, explore=self.explore)
            else:
                if actor is None:
                    init_action_mean = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.1])
                    init_action_cov = np.diag(np.array([0.3,0.3,0.3,0.3,0.3,0.05])**2)
                    sample_actions = np.random.multivariate_normal(init_action_mean, init_action_cov, size=(len(sublist_trans_obs), 256))
                else:
                    # slow method for now.
                    sample_actions = [actor.predict_batch(sess, *model_inputs, explore=True) for _ in range(256)]
                    sample_actions = np.array(sample_actions).transpose((1,0,2))
                sublist_trans_actions = critic.predict_batch_action(sess, *model_inputs,
                                             init_action_samples = sample_actions,
                                             iterations = 10, q_threshold=None)
            sublist_trans_actions = np.clip(sublist_trans_actions, self.env.action_low, self.env.action_high)
            idx = 0
            for i,k in enumerate(reward_keys):
                if k==key:
                    trans_actions[i]=sublist_trans_actions[idx]
                    idx += 1

        actions = []
        for tac, tf in zip(trans_actions, transforms):
            _, ac, _ = unifying_transform_decode(None, tac, None, tf)
            actions.append(ac)
        actions = np.array(actions)

        obs, rewards, dones, infos = self.env.step(actions)
        for ob_u, ac_u, r, ia, ia_u, key in zip(trans_obs, trans_actions, rewards,
                                                    intended_actions, trans_intended_actions, reward_keys):
            stats = self.model_stats_dict[key]
            reward = 1.0 if hash_dict(r) == hash_dict(ia) else 0.0
            stats.put(reward)
            if key in self.buffer_dict:
                self.buffer_dict[key].put(ob_u, ac_u, reward, ia_u)

        if self.eval_render:
            self.env.render()
        if self.eval_save:
            for ob,ia,r in zip(obs, intended_actions, rewards):
                if hash_dict(r) == hash_dict(ia):
                    np.savetxt('end_state_%d.txt'%(self.count), ob)
                    self.count += 1
        self.obs = self.env.reset()
