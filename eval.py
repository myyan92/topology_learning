import time
import os
import pdb
import tensorflow as tf
import numpy as np

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
                self.model_dict[key].fit(self.sess, obs, actions, rewards, rewards)
                self.steps_dict[key] += 1
                if (self.steps_dict[key] % self.log_interval == 0):
                    self.model_dict[key].save(self.sess, os.path.join(self.save_dir, 'models', 'model-%s'%(key)) , step=self.steps_dict[key])
                    stat_string = self.model_stat_dict[key].stat()
                    print(stat_string)

def eval(
    env,
    topology_key,
    eval_size=100,
    load_path=None):

    model = Model(topology_key)
    model.build(action_init=None)

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=16,
        intra_op_parallelism_threads=16)
    tf_config.gpu_options.allow_growth=True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    if load_path is not None:
        model.load(sess, load_path)

    # hack reduce gaussian std
    #gaussian_logstd = sess.run(model.gaussian_logstd)
    #sess.run(tf.assign(model.gaussian_logstd, gaussian_logstd-1.0))

    model_stat = ModelStats(model_name=topology_key,size=eval_size)
    runner = Runner(env, [model], [model_stat], [], gamma=0.99)

    while model_stat.num_in_buffer < eval_size:
        runner.run(sess)
        stat_string = model_stat.stat()
        print(stat_string)


if __name__ == "__main__":
    env = KnotEnv(parallel=60)
    eval(env, 'move-R1_left-1_sign-1', load_path='./test/models/model-move-R1_left-1_sign-1-1972')

