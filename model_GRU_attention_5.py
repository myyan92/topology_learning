import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from attention_layer import attention
import pdb

class Model:
    def __init__(self, reward_key):
        self.scope=reward_key

    def build(self, action_dim=6):

        with tf.variable_scope(self.scope):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None,64,3])
            self.over_seg_obs = tf.placeholder(dtype=tf.float32, shape=[None,None,3])
            self.over_seg_pos = tf.placeholder(dtype=tf.float32, shape=[None,None,1])
            self.over_seg_length = tf.placeholder(dtype=tf.int32, shape=[None,])
            self.under_seg_obs = tf.placeholder(dtype=tf.float32, shape=[None,None,3])
            self.under_seg_pos = tf.placeholder(dtype=tf.float32, shape=[None,None,1])
            self.under_seg_length = tf.placeholder(dtype=tf.int32, shape=[None,])

            self.mask_whole = tf.fill([tf.shape(self.input)[0], 64], True)
            self.mask_over = tf.sequence_mask(self.over_seg_length)
            self.mask_under = tf.sequence_mask(self.under_seg_length)

            self.mask_whole_f = tf.cast(tf.expand_dims(self.mask_whole, axis=-1), tf.float32)
            self.mask_over_f = tf.cast(tf.expand_dims(self.mask_over, axis=-1), tf.float32)
            self.mask_under_f = tf.cast(tf.expand_dims(self.mask_under, axis=-1), tf.float32)

            over_position_encoding = self.position_encoding(self.over_seg_pos, self.over_seg_length, self.mask_over_f)
            under_position_encoding = self.position_encoding(self.under_seg_pos, self.under_seg_length, self.mask_under_f)
            whole_pos = tf.linspace(0.0, 1.0, 64)
            whole_pos = tf.reshape(whole_pos, [1, 64, 1])
            whole_pos = tf.tile(whole_pos, [tf.shape(self.input)[0],1,1])
            whole_length = tf.ones_like(self.over_seg_length)*63
            whole_position_encoding = self.position_encoding(whole_pos, whole_length, self.mask_whole_f)

            self.feature_over_0 = tf.concat([self.over_seg_obs, over_position_encoding], axis=-1)
            self.feature_under_0 = tf.concat([self.under_seg_obs, under_position_encoding], axis=-1)
            self.feature_whole_0 = tf.concat([self.input, whole_position_encoding], axis=-1)

            self.feature_over_0 = tf.layers.dense(self.feature_over_0, 512, activation=None, use_bias=False)
            self.feature_under_0 = tf.layers.dense(self.feature_under_0, 512, activation=None, use_bias=False)
            self.feature_whole_0 = tf.layers.dense(self.feature_whole_0, 512, activation=None, use_bias=False)

            prev_over, prev_under, prev_whole = self.feature_over_0, self.feature_under_0, self.feature_whole_0
            for layer_idx in range(4):
                with tf.variable_scope("attention_%d" % layer_idx):
                    with tf.variable_scope("attention_whole_to_over"):
                        whole_to_over  = attention(prev_over, prev_whole, self.mask_over, self.mask_whole,
                                                   hidden_size=512, num_heads=2, attention_dropout=0.0, train=None)   
                    with tf.variable_scope("attention_over_to_over"):
                        over_to_over   = attention(prev_over, prev_over, self.mask_over, self.mask_over,
                                                   hidden_size=512, num_heads=2, attention_dropout=0.0, train=None) 
                    with tf.variable_scope("attention_under_to_over"):
                        under_to_over  = attention(prev_over, prev_under, self.mask_over, self.mask_under,
                                                   hidden_size=512, num_heads=2, attention_dropout=0.0, train=None)
                    with tf.variable_scope("attention_whole_to_under"):
                        whole_to_under = attention(prev_under, prev_whole, self.mask_under, self.mask_whole,
                                                   hidden_size=512, num_heads=2, attention_dropout=0.0, train=None)
                    with tf.variable_scope("attention_over_to_under"):
                        over_to_under  = attention(prev_under, prev_over, self.mask_under, self.mask_over,
                                                   hidden_size=512, num_heads=2, attention_dropout=0.0, train=None)
                    with tf.variable_scope("attention_under_to_under"):
                        under_to_under = attention(prev_under, prev_under, self.mask_under, self.mask_under,
                                                   hidden_size=512, num_heads=2, attention_dropout=0.0, train=None)
                    with tf.variable_scope("attention_whole_to_whole"):
                        whole_to_whole = attention(prev_whole, prev_whole, self.mask_whole, self.mask_whole,
                                                   hidden_size=512, num_heads=2, attention_dropout=0.0, train=None)
                    with tf.variable_scope("attention_over_to_whole"):
                        over_to_whole  = attention(prev_whole, prev_over, self.mask_whole, self.mask_over,
                                                   hidden_size=512, num_heads=2, attention_dropout=0.0, train=None)
                    with tf.variable_scope("attention_under_to_whole"):
                        under_to_whole = attention(prev_whole, prev_under, self.mask_whole, self.mask_under,
                                                    hidden_size=512, num_heads=2, attention_dropout=0.0, train=None)

                    next_over = prev_over + self.dense(whole_to_over, 'fc_w_o', 512, activation=None, scale=0.1) + \
                                            self.dense(under_to_over, 'fc_u_o', 512, activation=None, scale=0.1) + \
                                            self.dense(over_to_over,  'fc_o_o', 512, activation=None, scale=0.1)
                    next_over = tf.contrib.layers.layer_norm(next_over, begin_norm_axis=-1) 
                    next_over_1 = self.dense(next_over, 'fc_o1', 512, activation=tf.nn.relu)
                    next_over_1 = self.dense(next_over_1, 'fc_o2', 512, activation=None, scale=0.1)
                    next_over_1 = tf.contrib.layers.layer_norm(next_over_1+next_over, begin_norm_axis=-1)

                    next_under = prev_under + self.dense(whole_to_under, 'fc_w_u', 512, activation=None, scale=0.1) + \
                                              self.dense(under_to_under, 'fc_u_u', 512, activation=None, scale=0.1) + \
                                              self.dense(over_to_under,  'fc_o_u', 512, activation=None, scale=0.1)
                    next_under = tf.contrib.layers.layer_norm(next_under, begin_norm_axis=-1)
                    next_under_1 = self.dense(next_under, 'fc_u1', 512, activation=tf.nn.relu)
                    next_under_1 = self.dense(next_under_1, 'fc_u2', 512, activation=None, scale=0.1)
                    next_under_1 = tf.contrib.layers.layer_norm(next_under_1+next_under, begin_norm_axis=-1)

                    next_whole = prev_whole + self.dense(whole_to_whole, 'fc_w_w', 512, activation=None, scale=0.1) + \
                                              self.dense(under_to_whole, 'fc_u_w', 512, activation=None, scale=0.1) + \
                                              self.dense(over_to_whole,  'fc_o_w', 512, activation=None, scale=0.1)
                    next_whole = tf.contrib.layers.layer_norm(next_whole, begin_norm_axis=-1)
                    next_whole_1 = self.dense(next_whole, 'fc_w1', 512, activation=tf.nn.relu)
                    next_whole_1 = self.dense(next_whole_1, 'fc_w2', 512, activation=None, scale=0.1)
                    next_whole_1 = tf.contrib.layers.layer_norm(next_whole_1+next_whole, begin_norm_axis=-1)

                    prev_over, prev_under, prev_whole = next_over_1*self.mask_over_f, next_under_1*self.mask_under_f, next_whole_1


            self.feature_over_4, self.feature_under_4, self.feature_whole_4 = prev_over, prev_under, prev_whole

            # general action GMM
            fc1 = tf.layers.dense(self.feature_whole_4, 512, name='final_fc1', activation=tf.nn.relu, use_bias=False)
            fc2 = tf.layers.dense(fc1, 256, name='final_fc2', activation=tf.nn.relu, use_bias=False)
            fc3 = tf.layers.dense(fc2, 1, name='final_pick_logit', activation=None, use_bias=False)
            self.pick_logits = tf.maximum(fc3, -200)

            self.categorical = tfp.distributions.Categorical(logits=tf.squeeze(self.pick_logits, axis=-1))
            sample_node = self.categorical.sample()
            self.sample_node = tf.expand_dims(sample_node, -1) # to use gather
            ML_node = tf.argmax(self.pick_logits, axis=-1) # maximum likelyhood
            self.ML_node = tf.expand_dims(ML_node, -1)

            self.pick_point_input = tf.placeholder(dtype=tf.int32, shape=[None,1])

            self.action_feature = tf.layers.dense(fc1, 256, name='action_feature', activation=tf.nn.relu, use_bias=False)
            pick_point_action_feature = tf.gather(self.action_feature, self.pick_point_input, batch_dims=1, axis=1)
            pick_point_action_feature = tf.squeeze(pick_point_action_feature, axis=1)
            gaussian_mean_init = tf.constant_initializer([0.0,0.0,0.0,0.0,0.1])
            self.gaussian_mean = self.dense(pick_point_action_feature, 'gaussian_mean', action_dim-1, activation=None,
                                            scale=0.01, bias_init=gaussian_mean_init)
            self.gaussian_std = self.dense(pick_point_action_feature, 'gaussian_std', action_dim-1, activation=tf.nn.softplus,
                                            scale=0.01, bias_init=tf.constant_initializer([-1.5,-1.5,-1.5,-1.5,-2.0]))
            self.gaussian = tfp.distributions.MultivariateNormalDiag(loc=self.gaussian_mean,
                                                                     scale_diag = self.gaussian_std+0.001)

            self.action_first = tf.gather(whole_pos, self.pick_point_input, batch_dims=1, axis=1)
            self.action_first = tf.squeeze(self.action_first, axis=1)
            self.sample_action_second = self.gaussian.sample()
            self.sample_action = tf.concat([self.action_first, self.sample_action_second], axis=-1)
            self.ML_action = tf.concat([self.action_first, self.gaussian_mean], axis=-1)

            # state value
            state_value_feature = tf.reduce_sum(self.action_feature*tf.nn.softmax(self.pick_logits, axis=1), axis=1)
            fc1 = self.dense(state_value_feature, 'vf_fc1', 256, activation=tf.nn.relu)
            fc2 = self.dense(fc1, 'vf_fc2', 256, activation=tf.nn.relu)   
            self.state_value = self.dense(fc2, 'state_value', 1, activation=None)   

            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=500)

    def conv_layer(self, bottom, name, channels, kernel=3, stride=1, activation=tf.nn.relu):
        with tf.variable_scope(name):
            k_init = tf.variance_scaling_initializer()
            b_init = tf.zeros_initializer()
            output = tf.layers.conv1d(bottom, channels, kernel_size=kernel, strides=stride, padding='SAME',
                                      activation=activation, kernel_initializer=k_init, bias_initializer=b_init)
        return output

    def dense(self, bottom, name, channels, activation, scale=1.0, bias_init=None):
        with tf.variable_scope(name):
            k_init = tf.variance_scaling_initializer(scale)
            b_init = bias_init if bias_init is not None else tf.zeros_initializer()
            output = tf.layers.dense(bottom, channels, activation=activation,
                                     kernel_initializer=k_init, bias_initializer=b_init)
        return output

    def position_encoding(self, absolute_pos, lengths, mask):
        absolute_pos_sins = []
        for k in range(4):
            absolute_pos_sins.append( tf.sin(absolute_pos*k*3.1415) )
        relative_pos = ((absolute_pos - absolute_pos[:,0:1,:]) * 63
                         / tf.reshape(tf.cast(lengths, tf.float32)-1.0, [-1,1,1]))
        relative_pos_sins = []
        for k in range(4):
            relative_pos_sins.append( tf.sin(relative_pos*k*3.1415) )
        position_encoding = tf.concat([absolute_pos] + absolute_pos_sins + [relative_pos] + relative_pos_sins, axis=-1)
        position_encoding = position_encoding * mask
        return position_encoding

    def predict_single(self, sess, obs, over_seg_dict, under_seg_dict, explore=False):
        feed_dict = {self.input: obs[None],
                     self.over_seg_obs: over_seg_dict['obs'][None],
                     self.over_seg_pos: over_seg_dict['pos'][None],
                     self.over_seg_length: over_seg_dict['length'][None],
                     self.under_seg_obs: under_seg_dict['obs'][None],
                     self.under_seg_pos: under_seg_dict['pos'][None],
                     self.under_seg_length: under_seg_dict['length'][None]}
        if explore:
            node = sess.run(self.sample_node, feed_dict=feed_dict)
            feed_dict[self.pick_point_input] = node
            pred, = sess.run([self.sample_action], feed_dict=feed_dict)
        else:
            node = sess.run(self.ML_node, feed_dict=feed_dict)
            feed_dict[self.pick_point_input] = node
            pred, = sess.run([self.ML_action], feed_dict=feed_dict)
        return pred[0]

    def predict_batch(self, sess, obs, over_seg_dict, under_seg_dict, explore=False):
        feed_dict = {self.input: obs,
                     self.over_seg_obs: over_seg_dict['obs'],
                     self.over_seg_pos: over_seg_dict['pos'],
                     self.over_seg_length: over_seg_dict['length'],
                     self.under_seg_obs: under_seg_dict['obs'],
                     self.under_seg_pos: under_seg_dict['pos'],
                     self.under_seg_length: under_seg_dict['length']}
        if explore:
            node = sess.run(self.sample_node, feed_dict=feed_dict)
            feed_dict[self.pick_point_input] = node
            pred, = sess.run([self.sample_action], feed_dict=feed_dict)
        else:
            node = sess.run(self.ML_node, feed_dict=feed_dict)
            feed_dict[self.pick_point_input] = node
            pred, = sess.run([self.ML_action], feed_dict=feed_dict)
        return pred

    def predict_single_vf(self, sess, obs, over_seg_dict, under_seg_dict):
        feed_dict = {self.input: obs[None],
                     self.over_seg_obs: over_seg_dict['obs'][None],
                     self.over_seg_pos: over_seg_dict['pos'][None],
                     self.over_seg_length: over_seg_dict['length'][None],
                     self.under_seg_obs: under_seg_dict['obs'][None],
                     self.under_seg_pos: under_seg_dict['pos'][None],
                     self.under_seg_length: under_seg_dict['length'][None]}
        pred, = sess.run([self.state_value], feed_dict=feed_dict)
        return pred[0]

    def predict_batch_vf(self, sess, obs, over_seg_dict, under_seg_dict):
        feed_dict = {self.input: obs,
                     self.over_seg_obs: over_seg_dict['obs'],
                     self.over_seg_pos: over_seg_dict['pos'],
                     self.over_seg_length: over_seg_dict['length'],
                     self.under_seg_obs: under_seg_dict['obs'],
                     self.under_seg_pos: under_seg_dict['pos'],
                     self.under_seg_length: under_seg_dict['length']}
        pred, = sess.run([self.state_value], feed_dict=feed_dict)
        return pred

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def setup_optimizer(self, learning_rate, ent_coef, vf_coef, max_grad_norm):
        with tf.variable_scope(self.scope):

            self.train_node_input = tf.placeholder(tf.float32, [None,])
            self.train_action_second = tf.placeholder(tf.float32, [None, 5])
            self.advantage = tf.placeholder(tf.float32, [None])
            self.reward = tf.placeholder(tf.float32, [None])

            # Policy loss
            neglogpac = -self.categorical.log_prob(self.train_node_input) - self.gaussian.log_prob(self.train_action_second)
            self.pg_loss = tf.reduce_mean(self.advantage * neglogpac)
            # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
            self.entropy = tf.reduce_mean(self.gaussian.entropy()+self.categorical.entropy())
            # Value loss
            self.vf_loss = tf.reduce_mean((tf.squeeze(self.state_value) - self.reward)**2)
            self.loss = self.pg_loss - self.entropy*ent_coef + self.vf_loss*vf_coef

            params = self.get_trainable_variables()
            grads = tf.gradients(self.loss, params)
            if max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.optimizer = optimizer.apply_gradients(grads)

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('pg_loss', self.pg_loss)
            tf.summary.scalar('vf_loss', self.vf_loss)
            tf.summary.scalar('entropy', self.entropy)
            self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, obs, over_seg_dict, under_seg_dict, actions, advantages, rewards):
        nodes = actions[:,0]
        node_index = nodes*63
        legal_actions = np.all(node_index <= 63.0) and np.all(node_index >= 0.0)
        if not legal_actions:
            print("illegal actions")
            pdb.set_trace()
        feed_dict= {self.input:obs,
                    self.train_node_input:node_index.astype(np.int32),
                    self.train_action_second:actions[:,1:],
                    self.pick_point_input:node_index[:,np.newaxis].astype(np.int32),
                    self.advantage:advantages,
                    self.reward:rewards,
                    self.over_seg_obs: over_seg_dict['obs'],
                    self.over_seg_pos: over_seg_dict['pos'],
                    self.over_seg_length: over_seg_dict['length'],
                    self.under_seg_obs: under_seg_dict['obs'],
                    self.under_seg_pos: under_seg_dict['pos'],
                    self.under_seg_length: under_seg_dict['length']
        }
        loss, debug_softmax, debug_mean = sess.run([self.loss, self.pick_logits, self.gaussian_mean], feed_dict=feed_dict)
        valid_logits = debug_softmax.flatten()
        valid_logits = valid_logits[valid_logits>-300]
        if np.any(np.isnan(debug_softmax)) or np.any(np.isnan(debug_mean)) or np.isnan(loss):
            print("loss is nan")
            pdb.set_trace()
        elif np.mean(valid_logits) <= -190:
            print("pick logits collapsing")
            pdb.set_trace()
        else:
           sess.run(self.optimizer, feed_dict=feed_dict)
        return loss

    def save(self, sess, file_dir, step):
        self.saver.save(sess, file_dir, global_step=step)

    def load(self, sess, snapshot):
        self.saver.restore(sess, snapshot)


if __name__=="__main__":
    model=Model('R1')
    model.build()
    model.setup_optimizer(1e-3, 1e-3, 1e-3, 1e-2)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    states = np.zeros((4,64,3))
    states[:,:,0]=np.linspace(0,1,64)
    intersect = np.random.randint(5,60, size=(4,2))
    intersect = [np.amin(intersect, axis=1), np.amax(intersect, axis=1)]
    over_max = np.amax(intersect[0])
    over_seg=np.zeros((4,over_max+1,3))
    over_pos=np.zeros((4,over_max+1,1))
    over_length=intersect[0]+1
    for i in range(4):
        over_seg[i,:intersect[0][i]+1,:]=states[i,:intersect[0][i]+1,:]
        over_pos[i,:intersect[0][i]+1,0]=np.arange(0, intersect[0][i]+1)
    under_max = 64-np.amin(intersect[1])
    under_seg=np.zeros((4,under_max,3))
    under_pos=np.ones((4,under_max,1))
    under_length=64-intersect[1]
    for i in range(4):
        under_seg[i,:64-intersect[1][i],:]=states[i,intersect[1][i]:64,:]
        under_pos[i,:64-intersect[1][i],0]=np.arange(intersect[1][i], 64)
    action = model.predict_batch(sess, states, over_seg_dict={'obs':over_seg, 'pos':over_pos/63.0, 'length':over_length},
                                               under_seg_dict={'obs':under_seg, 'pos':under_pos/63.0, 'length':under_length}, explore=True)
    print(action.shape)

    model.fit(sess, states, over_seg_dict={'obs':over_seg, 'pos':over_pos/63.0, 'length':over_length},
                            under_seg_dict={'obs':under_seg, 'pos':under_pos/63.0, 'length':under_length},
                            actions=action, advantages=np.ones(4,), rewards=np.ones(4,))
