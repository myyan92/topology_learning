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
import gin


class A2C():
    def __init__(self, models, model_stats, buffers, log_interval,
                 train_batch_size, replay_start, replay_grow, save_dir):
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
        self.save_dir = save_dir
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
                #print(self.steps_dict[key])
                if (self.steps_dict[key] % self.log_interval == 0):
                    self.model_dict[key].save(self.sess, os.path.join(self.save_dir, 'models', 'model-%s'%(key)) , step=self.steps_dict[key])
                    stat_string = self.model_stat_dict[key].stat()
                    print(stat_string)

@gin.configurable
def learn(
    env,
    reward_keys,
    pretrain_buffers,
    total_timesteps=int(80e6),
    train_batch_size=32,
    vf_coef=0.5,
    ent_coef=0.001, # was 0.01 before debuging 2to3.
    max_grad_norm=0.5,
    lr=7e-4,
    gamma=0.99,
    log_interval=10,
    save_dir='./test'):

    models = [Model(key) for key in reward_keys]
    for model in models:
        model.build()
        model.setup_optimizer(learning_rate=lr, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)

    buffers = [Buffer(reward_key=key, size=50000) for key in reward_keys]
    for buffer, init_buffer in zip(buffers, pretrain_buffers):
        buffer.load(init_buffer)

    model_stats = [ModelStats(model_name=key) for key in reward_keys]

    # pretrain
    a2c = A2C(models, model_stats, buffers, log_interval, train_batch_size, replay_start=4, replay_grow=1, save_dir=save_dir)
    a2c.update()

    buffers = [Buffer(reward_key=key, size=50000, filter_success=False) for key in reward_keys] # re-init buffers
    a2c.buffer_dict = {buffer.reward_key:buffer for buffer in buffers}
    runner = Runner(env, models, model_stats, buffers, gamma=gamma)

    def signal_handler(sig, frame):
        for buffer in buffers:
             buffer.dump()
             print('dump big buffer succeed! Size:', buffer.num_in_buffer)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    for _ in range(total_timesteps):
        runner.run(a2c.sess)
        a2c.update()


if __name__ == "__main__":

    gin_config_file = sys.argv[1]
    gin.parse_config_file(gin_config_file)
    env = KnotEnv(parallel=64)
    learn(env)

