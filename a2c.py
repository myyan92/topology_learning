import time
import os
import pdb
import tensorflow as tf
import numpy as np
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
from knot_env import KnotEnv
from runner import Runner
from buffer import Buffer
from model_GRU import Model
from model_stats import ModelStats

class A2C():
    def __init__(self, models, model_stats, buffers, log_interval,
                 train_batch_size, replay_start, replay_grow):
        self.model_dict = {model.scope:model for model in models}
        self.model_stat_dict = {model_stat.model_name:model_stat for model_stat in model_stats}
        self.steps_dict = {model.scope:0 for model in models}
        self.buffer_dict = {buffer.reward_key:buffer for buffer in buffers}
        assert set(self.model_dict.keys()) == set(self.buffer_dict.keys())

        self.log_interval = log_interval
        self.train_batch_size = train_batch_size
        self.replay_start = replay_start
        self.replay_grow = replay_grow
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tf_config)
        self.save_dir = 'test'
        if not os.path.exists(os.path.join(self.save_dir, 'models/')):
            os.makedirs(os.path.join(self.save_dir, 'models/'))
        self.train_writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'tfboard'), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())


    def update(self):
        for key in self.buffer_dict:
            while self.buffer_dict[key].has_atleast(self.replay_start+self.replay_grow*self.steps_dict[key]):
                obs, actions, rewards = self.buffer_dict[key].get(self.train_batch_size)
                # add augmentation
                obs, actions = self.buffer_dict[key].augment(obs, actions)
                self.model_dict[key].fit(self.sess, obs, actions, rewards, rewards)
                self.steps_dict[key] += 1
                if (self.steps_dict[key] % self.log_interval == 0):
                    self.model_dict[key].save(self.sess, os.path.join(self.save_dir, 'models', 'model-%s'%(key)) , step=self.steps_dict[key])
                    stat_string = self.model_stat_dict[key].stat()
                    print(stat_string)

def learn(
    env,
    total_timesteps=int(80e6),
    train_batch_size=32,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    gamma=0.99,
    log_interval=4,
    load_path=None):

    # Instantiate the model object (that creates step_model and train_model)
    keys = ['move-R1_left-1_sign-1',
#            'move-R1_left-1_sign--1',
#            'move-R1_left--1_sign-1',
#            'move-R1_left--1_sign--1',
#            'move-R2_left-1_over_before_under-1',
#            'move-R2_left-1_over_before_under--1',
#            'move-R2_left--1_over_before_under-1',
#            'move-R2_left--1_over_before_under--1'
           ]

    gaussian_init = {'move-R1_left-1_sign-1': np.array([0.9,0.0,0.0,0.0,0.0,0.1]),  # for all trivial state
#                     'move-R1_left-1_sign-1': np.array([0.9,0.0,0.2,-0.2,-0.2,0.1]), # for fixed straight state
#                     'move-R1_left-1_sign--1': np.array([0.1,0.0,0.2,0.2,-0.2,0.1]),
#                     'move-R1_left--1_sign-1': np.array([0.1,0.0,-0.2,0.2,0.2,0.1]),
#                     'move-R1_left--1_sign--1': np.array([0.9,0.0,-0.2,-0.2,0.2,0.1]),
#                     'move-R2_left-1_over_before_under-1': np.array([0.3,0.0,0.2,0.2,-0.2,0.1]),
#                     'move-R2_left-1_over_before_under--1': np.array([0.7,0.0,0.2,-0.2,-0.2,0.1]),
#                     'move-R2_left--1_over_before_under-1': np.array([0.3,0.0,-0.2,0.2,0.2,0.1]),
#                     'move-R2_left--1_over_before_under--1': np.array([0.7,0.0,-0.2,-0.2,0.2,0.1])
                     }
    models = [Model(key) for key in keys]
    for model in models:
        model.build(action_init=gaussian_init[model.scope])
        model.setup_optimizer(learning_rate=lr, ent_coef=ent_coef, max_grad_norm=max_grad_norm)
    #if load_path is not None:
    #    model.load(load_path)

    buffers = [Buffer(reward_key=key, size=50000) for key in keys]
    for buffer in buffers:
        buffer.load(buffer.reward_key+'_buffer.npz')

    model_stats = [ModelStats(model_name=key) for key in keys]
    runner = Runner(env, models, model_stats, buffers, gamma=gamma)
    a2c = A2C(models, model_stats, buffers, log_interval, train_batch_size, replay_start=4, replay_grow=0.3)

    for _ in range(total_timesteps):
        a2c.update()
        runner.run(a2c.sess)


if __name__ == "__main__":
    env = KnotEnv(parallel=60)
    learn(env)

