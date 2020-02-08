import numpy as np
from planner import encode
from model_GRU_attention import Model
from state_encoder import unifying_transform_encode, unifying_transform_decode
import pdb

def random_uniform_sampler():
    def sampling_func(obs):
        return np.random.uniform(low=[0.0,-0.5,-0.5,-0.5,-0.5,0.02],
                                 high=[1.0, 0.5, 0.5, 0.5, 0.5, 0.2])
    return sampling_func

def random_gaussian_heuristic_sampler(gaussian_mean, gaussian_std):
    def sampling_func(obs):
        return np.random.normal(loc=gaussian_mean, scale=gaussian_std)
    return sampling_func

def model_sampler(sess, actor_model, critic_model, intended_action):
    def sampling_func(obs):
        obs_u, _, ia_u, transform = unifying_transform_encode(obs, None, intended_action)
        model_inputs = encode([obs_u], [ia_u])
        if critic_model is None:
            action_u = actor_model.predict_batch(sess, *model_inputs, explore=True)
        else:
            if actor_model is None:
                init_action_mean = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.1])
                init_action_cov = np.diag(np.array([0.3,0.3,0.3,0.3,0.3,0.05])**2)
                sample_actions = np.random.multivariate_normal(init_action_mean, init_action_cov, size=(1,256))
                action_u = critic_model.predict_batch_action(sess, *model_inputs,
                                                         init_action_samples = sample_actions,
                                                         iterations = 10, q_threshold=None)
            else:
                # slow method for now.
                sample_actions = [actor_model.predict_batch(sess, *model_inputs, explore=True) for _ in range(256)]
                sample_actions = np.array(sample_actions).transpose((1,0,2))
                action_u = critic_model.predict_batch_action(sess, *model_inputs,
                                                         init_action_samples = sample_actions,
                                                         iterations = 1, q_threshold=None)
        _, action, _ = unifying_transform_decode(None, action_u[0], None, transform)
        return action
    return sampling_func
