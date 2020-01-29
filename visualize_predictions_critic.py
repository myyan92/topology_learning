import time
import os
import pdb
import tensorflow as tf
import numpy as np

from model_GRU_attention_C3 import Model

from planner import encode
from state_encoder import unifying_transform_encode, unifying_transform_decode
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def visualize(
    topology_key,
    load_path=None):

    model = Model(topology_key)
    #model.build(action_init=None)
    model.build()

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=16,
        intra_op_parallelism_threads=16)
    tf_config.gpu_options.allow_growth=True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    if load_path is not None:
        model.load(sess, load_path)

    # load states
    states = []
    files = glob.glob('1loop_states/012.txt')
    files.sort()
    for f in files:
        states.append(np.loadtxt(f))
#    states = [state*np.array([-1.0,-1.0,1.0]) for state in states]
    intended_action = {'move':'cross', 'over_idx':0, 'under_idx':1, 'sign':1}
    trans_obs = []
    for st in states:
        obs_u, _, trans_intended_action, transform = unifying_transform_encode(st, None, intended_action)
        trans_obs.append(obs_u)

    trans_obs = trans_obs * 400 # hack
    model_inputs = encode(trans_obs, [trans_intended_action]*len(states)*400)

    action_mean = np.array([0.01, 0.18,0.0,-0.05,0.0, 0.1])
#    action_std = np.array([0.02, 0.05,0.05,0.05,0.05, 0.02])
#    actions = np.random.normal(loc=action_mean, scale=action_std, size=(200,6))
#    actions = np.clip(actions, np.array([0.0,-0.5,-0.5,-0.5,-0.5,0.02]), np.array([1.0,0.5,0.5,0.5,0.5,0.2]))
    actions = np.tile(action_mean, (400,1))
    x=np.linspace(-0.15, 0.05, 20)
    y=np.linspace(-0.1, 0.1, 20)
    x,y=np.meshgrid(x,y)
    actions[:,3]=x.flatten()
    actions[:,4]=y.flatten()
    qs, vs = model.predict_batch(sess, *model_inputs, actions)
    qs = 1/(1 + np.exp(-qs)) # sigmoid
    print(np.amin(qs), np.amax(qs))

    # plotting
    plt.plot(trans_obs[0][:,0], trans_obs[0][:,1])
    plt.scatter(actions[:,3], actions[:,4], c=qs[:,0])
#    plt.savefig(f.replace('1loop_states/', 'visualize/').replace('.txt', '.png'))
#    plt.close()
    plt.show()

if __name__ == "__main__":
#    visualize('move-cross_endpoint-under_sign-1', load_path='./2to3-cross-endpointunder-sign1-randstate-GMM_layernorm/models/model-move-cross_endpoint-under_sign-1-9900')
#    visualize('move-cross_endpoint-over_sign-1', load_path='./1to2-cross-endpointover-sign1-randstate/models/model-move-cross_endpoint-over_sign-1-1800')
    visualize('move-cross_endpoint-over_sign-1', load_path='./1to2-cross-endpointover-sign1-randstate_mC5/models/model-move-cross_endpoint-over_sign-1-5750')
#    visualize('move-R1_left-1_sign-1', load_path='./0to1-R1-left1-sign1-augstart/models/model-move-R1_left-1_sign-1-1860')
#    visualize('move-R2_left-1_over_before_under-1', load_path = './0to1-R2-left1-obu1-augstart/models/model-move-R2_left-1_over_before_under-1-6740')
