import numpy as np
from planner import encode
from model_GRU_attention import Model

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
    # TODO use unifying encoder and decoder
    def sampling_func(obs):
        obs, over_seg_dict, under_seg_dict = encode(obs, intended_action)
        action = model.predict_single(sess, obs, over_seg_dict, under_seg_dict, explore=True)
        return action
    return sampling_func
