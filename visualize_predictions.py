import time
import os
import pdb
import tensorflow as tf
import numpy as np

#from model_GRU import Model
from model_GRU_attention_5 import Model

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

    # hack reduce gaussian std
    #gaussian_logstd = sess.run(model.gaussian_logstd)
    #sess.run(tf.assign(model.gaussian_logstd, gaussian_logstd-1.0))
    #vars = model.get_trainable_variables()
    #vars = [v for v in vars if 'gaussian_logstd' in v.name]
    #vars = [v for v in vars if 'bias' in v.name]
    #val = sess.run(vars[0])
    #sess.run(tf.assign(vars[0], val-2.0))

    # load states
    states = []
    files = glob.glob('1loop_states/???.txt')
    files.sort()
    for f in files:
        states.append(np.loadtxt(f))
#    states = [state*np.array([-1.0,-1.0,1.0]) for state in states]
    intended_action = {'move':'cross', 'over_idx':0, 'under_idx':1, 'sign':1}
    trans_obs = []
    for st in states:
        obs_u, _, trans_intended_action, transform = unifying_transform_encode(st, None, intended_action)
        trans_obs.append(obs_u)
    model_inputs = encode(trans_obs, [trans_intended_action]*len(states))
    state_values = model.predict_batch_vf(sess, *model_inputs)
    actions = model.predict_batch(sess, *model_inputs) # only for model v2
    pdb.set_trace()
    action_nodes = (actions[:,0]-model_inputs[1]['pos'][:,0,0])*63
    feed_dict = {model.input: model_inputs[0],
                 model.over_seg_obs: model_inputs[1]['obs'],
                 model.over_seg_pos: model_inputs[1]['pos'],
                 model.over_seg_length: model_inputs[1]['length'],
                 model.under_seg_obs: model_inputs[2]['obs'],
                 model.under_seg_pos: model_inputs[2]['pos'],
                 model.under_seg_length: model_inputs[2]['length'],
                 model.pick_point_input: action_nodes.astype(np.int32)[:,np.newaxis]}

    pick_probs, gaussian_means, gaussian_tril = sess.run([model.categorical.probs, model.gaussian_mean, model.gaussian_tril], feed_dict=feed_dict)
    gaussian_stds = np.diagonal(gaussian_tril, axis1=1, axis2=2)

    # saving
    with open('visualize/state_values.txt', 'w') as fout:
        for vf, f in zip(state_values, files):
            fout.write('%s: %f\n'%(f, vf))
    pdb.set_trace()
    # plotting
    for f,st,vf,pp,mu,std in zip(files, trans_obs, state_values, pick_probs, gaussian_means, gaussian_stds):
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        ax.scatter(st[:,0],st[:,1], c=np.arange(64), label='value=%f'%(vf))
        # plot pick probs?
        ellipse1 = Ellipse(xy=mu[0:2], width=std[0], height=std[1], fc='C2', alpha=0.3)
        ax.add_artist(ellipse1)
        ellipse2 = Ellipse(xy=mu[2:4], width=std[2], height=std[3], fc='C3', alpha=0.3)
        ax.add_artist(ellipse2)
        plt.legend()
        plt.savefig(f.replace('1loop_states/', 'visualize/').replace('.txt', '.png'))
        plt.close()

if __name__ == "__main__":
#    visualize('move-cross_endpoint-under_sign-1', load_path='./2to3-cross-endpointunder-sign1-randstate-GMM_layernorm/models/model-move-cross_endpoint-under_sign-1-9900')
#    visualize('move-cross_endpoint-over_sign-1', load_path='./1to2-cross-endpointover-sign1-randstate/models/model-move-cross_endpoint-over_sign-1-1800')
    visualize('move-cross_endpoint-over_sign-1', load_path='./1to2-cross-endpointover-sign1-randstate_m5/models/model-move-cross_endpoint-over_sign-1-660')
#    visualize('move-R1_left-1_sign-1', load_path='./0to1-R1-left1-sign1-augstart/models/model-move-R1_left-1_sign-1-1860')
#    visualize('move-R2_left-1_over_before_under-1', load_path = './0to1-R2-left1-obu1-augstart/models/model-move-R2_left-1_over_before_under-1-6740')
