import numpy as np
from planner import encode
from model_GRU_attention import Model
from state_encoder import unifying_transform_encode, unifying_transform_decode

def random_uniform_sampler():
    def sampling_func(obs):
        return np.random.uniform(low=[0.0,-0.5,-0.5,-0.5,-0.5,0.02],
                                 high=[1.0, 0.5, 0.5, 0.5, 0.5, 0.2])
    return sampling_func

def random_gaussian_heuristic_sampler(gaussian_mean, gaussian_std):
    def sampling_func(obs):
        return np.random.normal(loc=gaussian_mean, scale=gaussian_std)
    return sampling_func

def model_sampler(sess, model, intended_action):
    def sampling_func(obs):
        obs_u, _, ia_u, transform = unifying_transform_encode(obs, None, intended_action)
        obs_u, over_seg_dict_u, under_seg_dict_u = encode([obs_u], [ia_u])
        action_u = model.predict_batch(sess, obs_u, over_seg_dict_u, under_seg_dict_u, explore=True)
        _, action, _ = unifying_transform_decode(None, action_u[0], None, transform)
        return action
    return sampling_func
