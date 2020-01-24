import time
import os, sys
import pdb
import tensorflow as tf
import numpy as np

from knot_env import KnotEnv
from advanced_runner import Runner
from advanced_buffer import Buffer
from model_GRU_attention_5 import Model
from model_stats import ModelStats
import gin

@gin.configurable
def eval(
    env,
    model_key,
    eval_size=56,
    load_path=None):

    model = Model(model_key)
    model.build()

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
    #vars = model.get_trainable_variables()
    #vars = [v for v in vars if 'gaussian_logstd' in v.name]
    #vars = [v for v in vars if 'bias' in v.name]
    #val = sess.run(vars[0])
    #sess.run(tf.assign(vars[0], val-2.0))

    model_stat = ModelStats(model_name=model_key,size=eval_size)
    runner = Runner(env, [model], [model_stat], [], gamma=0.99)

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
