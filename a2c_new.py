import time
import os, sys
import signal
import pdb
import tensorflow as tf
import numpy as np
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
from knot_env import KnotEnv
from advanced_runner import Runner
from advanced_buffer import Buffer
from model_GRU_attention import Model
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
                obs, actions, rewards, over_seg_dict, under_seg_dict = self.buffer_dict[key].get(self.train_batch_size)
                # add augmentation
                obs, actions, over_seg_dict, under_seg_dict = self.buffer_dict[key].augment(
                                                                   obs, actions, over_seg_dict, under_seg_dict)
                self.model_dict[key].fit(self.sess, obs, over_seg_dict, under_seg_dict, actions, rewards, rewards)
                self.steps_dict[key] += 1
                print(self.steps_dict[key])
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
    keys = ['move-cross_endpoint-over_sign-1',
            'move-cross_endpoint-over_sign--1',
            'move-R1_left-1_sign-1',
            'move-R1_left-1_sign--1',
            'move-R1_left--1_sign-1',
            'move-R1_left--1_sign--1',
            'move-R2_left-1_over_before_under-1',
            'move-R2_left-1_over_before_under--1',
            'move-R2_left--1_over_before_under-1',
            'move-R2_left--1_over_before_under--1',
            'move-R2_left-1_diff',
            'move-R2_left--1_diff',
           ]
    init_buffer_names = [
                        ['move-cross_over_idx-0_sign-1_under_idx-1_buffer.npz',
                         'move-cross_over_idx-0_sign-1_under_idx-2_buffer.npz',
                         'move-cross_over_idx-2_sign-1_under_idx-0_buffer.npz',
                         'move-cross_over_idx-2_sign-1_under_idx-1_buffer.npz'],
                        ['move-cross_over_idx-0_sign--1_under_idx-1_buffer.npz',
                         'move-cross_over_idx-0_sign--1_under_idx-2_buffer.npz',
                         'move-cross_over_idx-2_sign--1_under_idx-0_buffer.npz',
                         'move-cross_over_idx-2_sign--1_under_idx-1_buffer.npz'],
                        ['idx-1_left-1_move-R1_sign-1_buffer.npz',
                         'idx-2_left-1_move-R1_sign-1_buffer.npz'],
                        ['idx-0_left-1_move-R1_sign--1_buffer.npz',
                         'idx-1_left-1_move-R1_sign--1_buffer.npz'],
                        ['idx-0_left--1_move-R1_sign-1_buffer.npz',
                         'idx-1_left--1_move-R1_sign-1_buffer.npz'],
                        ['idx-1_left--1_move-R1_sign--1_buffer.npz',
                         'idx-2_left--1_move-R1_sign--1_buffer.npz'],
                        ['left-1_move-R2_over_before_under-1_over_idx-0_under_idx-0_buffer.npz',
                         'left-1_move-R2_over_before_under-1_over_idx-1_under_idx-1_buffer.npz'],
                        ['left-1_move-R2_over_before_under--1_over_idx-1_under_idx-1_buffer.npz',
                         'left-1_move-R2_over_before_under--1_over_idx-2_under_idx-2_buffer.npz'],
                        ['left--1_move-R2_over_before_under-1_over_idx-0_under_idx-0_buffer.npz',
                         'left--1_move-R2_over_before_under-1_over_idx-1_under_idx-1_buffer.npz'],
                        ['left--1_move-R2_over_before_under--1_over_idx-1_under_idx-1_buffer.npz',
                         'left--1_move-R2_over_before_under--1_over_idx-2_under_idx-2_buffer.npz'],
                        ['left-1_move-R2_over_before_under--1_over_idx-1_under_idx-0_buffer.npz',
                         'left-1_move-R2_over_before_under--1_over_idx-2_under_idx-0_buffer.npz',
                         'left-1_move-R2_over_before_under--1_over_idx-2_under_idx-1_buffer.npz',
                         'left-1_move-R2_over_before_under-1_over_idx-0_under_idx-1_buffer.npz',
                         'left-1_move-R2_over_before_under-1_over_idx-0_under_idx-2_buffer.npz',
                         'left-1_move-R2_over_before_under-1_over_idx-1_under_idx-2_buffer.npz'],
                        ['left--1_move-R2_over_before_under--1_over_idx-1_under_idx-0_buffer.npz',
                         'left--1_move-R2_over_before_under--1_over_idx-2_under_idx-0_buffer.npz',
                         'left--1_move-R2_over_before_under--1_over_idx-2_under_idx-1_buffer.npz',
                         'left--1_move-R2_over_before_under-1_over_idx-0_under_idx-1_buffer.npz',
                         'left--1_move-R2_over_before_under-1_over_idx-0_under_idx-2_buffer.npz',
                         'left--1_move-R2_over_before_under-1_over_idx-1_under_idx-2_buffer.npz']
                       ]

    models = [Model(key) for key in keys]
    for model in models:
        model.build()
        model.setup_optimizer(learning_rate=lr, ent_coef=ent_coef, max_grad_norm=max_grad_norm)
    #if load_path is not None:
    #    model.load(load_path)

    buffers = [Buffer(reward_key=key, size=50000) for key in keys]
    for buffer, init_bufs in zip(buffers, init_buffer_names):
        for ib in init_bufs:
            buffer.append(ib)

    model_stats = [ModelStats(model_name=key) for key in keys]
    runner = Runner(env, models, model_stats, buffers, gamma=gamma)
    a2c = A2C(models, model_stats, buffers, log_interval, train_batch_size, replay_start=4, replay_grow=3)

    def signal_handler(sig, frame):
        for buffer in buffers:
             buffer.dump()
             print('dump big buffer succeed! Size:', buffer.num_in_buffer)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    for _ in range(total_timesteps):
        a2c.update()
        runner.run(a2c.sess)


if __name__ == "__main__":
    env = KnotEnv(parallel=90)
    learn(env)

