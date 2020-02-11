import time
import os, sys
import pdb
import tensorflow as tf
import numpy as np

from knot_env import KnotEnv
from advanced_runner import Runner
from advanced_buffer import Buffer
from model_GRU_attention_3 import Model as Model_Actor
from model_GRU_C3 import Model as Model_Critic
from model_stats import ModelStats
import gin

@gin.configurable
def eval(
    env,
    model_key,
    actor=True, critic=True,
    eval_size=192,
    actor_load_path=None, critic_load_path=None):

    if actor:
        actor_model = Model_Actor(model_key+'_actor')
        actor_model.build()
    if critic:
        critic_model = Model_Critic(model_key+'_critic')
        critic_model.build()

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=16,
        intra_op_parallelism_threads=16)
    tf_config.gpu_options.allow_growth=True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    if actor and actor_load_path is not None:
        vars_to_load = actor_model.get_trainable_variables()
        reader = tf.train.load_checkpoint(actor_load_path)
        for var in vars_to_load:
            new_name = var.name
            old_name = new_name.replace(model_key+'_actor', model_key)
            if old_name.endswith(':0'):
                old_name = old_name[:-2]
            tensor = reader.get_tensor(old_name)
            sess.run(tf.assign(var, tensor))
#        actor_model.load(sess, actor_load_path)
    if critic and critic_load_path is not None:
        vars_to_load = critic_model.get_trainable_variables()
        reader = tf.train.load_checkpoint(critic_load_path)
        for var in vars_to_load:
            new_name = var.name
            old_name = new_name.replace(model_key+'_critic', model_key)
            if old_name.endswith(':0'):
                old_name = old_name[:-2]
            tensor = reader.get_tensor(old_name)
            sess.run(tf.assign(var, tensor))
#        critic_model.load(sess, critic_load_path)

    model_stat = ModelStats(model_name=model_key,size=eval_size)
    runner = Runner(env, actor_models=[actor_model] if actor else [],
                         critic_models = [critic_model] if critic else [],
                         model_stats = [model_stat], buffers = [], gamma=0.99)

    while model_stat.num_in_buffer < eval_size:
        runner.run(sess)
        stat_string = model_stat.stat()
        print(stat_string)
        print(model_stat.reward[:model_stat.num_in_buffer])

if __name__ == "__main__":

    gin_config_file = sys.argv[1]
    gin.parse_config_file(gin_config_file)
    env = KnotEnv(parallel=8)
    eval(env)
