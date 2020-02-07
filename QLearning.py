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
from model_GRU_C3 import Model as Model_v3
from model_GRU_attention_C5 import Model as Model_v5
from model_stats import ModelStats
import matplotlib.pyplot as plt
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
                obs, actions, rewards, over_seg_dict, under_seg_dict = self.buffer_dict[key].get(self.train_batch_size//2)
                # add augmentation
                obs, actions, over_seg_dict, under_seg_dict = self.buffer_dict[key].augment(
                                                                   obs, actions, over_seg_dict, under_seg_dict)
                # make up random bad samples.
                fake_actions = np.random.uniform(low=np.array([0.0,-0.5,-0.5,-0.5,-0.5,0.02]),
                                                 high=np.array([1.0,0.5,0.5,0.5,0.5,0.2]), size=(self.train_batch_size//2, 6))
                # negative mining
                mining_actions = self.model_dict[key].predict_batch_action(self.sess, obs, over_seg_dict, under_seg_dict)
                # mixing
                fake_actions[0::2] = mining_actions[0::2]
                obs = np.concatenate([obs,obs], axis=0)
                actions = np.concatenate([actions, fake_actions], axis=0)
                over_seg_dict = {key:np.concatenate([val, val], axis=0) for key,val in over_seg_dict.items()}
                under_seg_dict = {key:np.concatenate([val, val], axis=0) for key,val in under_seg_dict.items()}
                rewards = np.concatenate([rewards, np.zeros(len(fake_actions))], axis=0)
                q_loss = self.model_dict[key].fit_q(self.sess, obs, over_seg_dict, under_seg_dict, actions, rewards[:,np.newaxis])
                print(q_loss)
                self.steps_dict[key] += 1
                if (self.steps_dict[key] % self.log_interval == 0):
                    self.model_dict[key].save(self.sess, os.path.join(self.save_dir, 'models', 'model-%s'%(key)) , step=self.steps_dict[key])
                    stat_string = self.model_stat_dict[key].stat()
                    print(stat_string)

@gin.configurable
def learn(
    env,
    reward_keys,
    model_type,
    pretrain_buffers,
    total_timesteps=int(80e6),
    train_batch_size=128,
    gamma=0.99,
    lr=1e-4,
    log_interval=50,
    save_dir='./test'):

    if model_type=='Model_v3':
        models = [Model_v3(key) for key in reward_keys]
    if model_type=='Model_v5':
        models = [Model_v5(key) for key in reward_keys]

    for model in models:
        model.build()
        model.setup_optimizer(learning_rate=lr)

    buffers = [Buffer(reward_key=key, size=50000, filter_success=False) for key in reward_keys]
    for buffer, init_buffer in zip(buffers, pretrain_buffers):
        buffer.load(init_buffer)

    model_stats = [ModelStats(model_name=key) for key in reward_keys]

    # pretrain
    a2c = A2C(models, model_stats, buffers, log_interval, train_batch_size, replay_start=32, replay_grow=0.05, save_dir=save_dir)
#    for model in models:
#        model.load(a2c.sess, './1to2-cross-endpointover-sign1-randstate_mC6_QT/models/model-move-cross_endpoint-over_sign-1-42000')
#        a2c.steps_dict[model.scope]=42000
    a2c.update()
    runner = Runner(env, models, model_stats, buffers, gamma=gamma)

    def signal_handler(sig, frame):
        for buffer in buffers:
             buffer.dump(path=save_dir)
             print('dump big buffer succeed! Size:', buffer.num_in_buffer)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        for _ in range(total_timesteps):
            runner.run(a2c.sess)
            a2c.update()
    except:
        raise
    finally:
        for buffer in buffers:
            buffer.dump(path=save_dir)

if __name__ == "__main__":

    model_type = sys.argv[1]
    gin_config_file = sys.argv[2]
    gin.parse_config_file(gin_config_file)
    env = KnotEnv(parallel=32)
    learn(env, model_type=model_type)

