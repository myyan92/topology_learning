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
            self.action = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])


            cell = tf.nn.rnn_cell.GRUCell(256, activation=tf.nn.tanh, name='gru_cell_whole')
            # TODO maybe share cell weights?
            self.biLSTM_whole, self.biLSTM_whole_final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell, self.input,
                                                                                   dtype = tf.float32, time_major=False)
            self.feature_whole = tf.concat([self.biLSTM_whole[0], self.biLSTM_whole[1], self.input], axis=2)
            self.mask_whole = tf.fill([tf.shape(self.input)[0], 64], True)

            cell = tf.nn.rnn_cell.GRUCell(256, activation=tf.nn.tanh, name='gru_cell_over')
            self.biLSTM_over, self.biLSTM_over_final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell,
                                                                                   self.over_seg_obs, self.over_seg_length,
                                                                                   dtype = tf.float32, time_major=False)
            self.feature_over = tf.concat([self.biLSTM_over[0], self.biLSTM_over[1], self.over_seg_obs], axis=2)
            self.mask_over = tf.sequence_mask(self.over_seg_length)

            cell = tf.nn.rnn_cell.GRUCell(256, activation=tf.nn.tanh, name='gru_cell_under')
            self.biLSTM_under, self.biLSTM_under_final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell,
                                                                                   self.under_seg_obs, self.under_seg_length,
                                                                                   dtype = tf.float32, time_major=False)
            self.feature_under = tf.concat([self.biLSTM_under[0], self.biLSTM_under[1], self.under_seg_obs], axis=2)
            self.mask_under = tf.sequence_mask(self.under_seg_length)

            # generating attention
            # padded parts of feature_over and feature_under are all zeros.
            with tf.variable_scope("attention_whole_to_over"):
                self.whole_to_over = attention(self.feature_over, self.feature_whole, self.mask_over, self.mask_whole,
                                               hidden_size=256, num_heads=2, attention_dropout=0.0, train=None)
            with tf.variable_scope("attention_whole_to_under"):
                self.whole_to_under = attention(self.feature_under, self.feature_whole, self.mask_under, self.mask_whole,
                                                hidden_size=256, num_heads=2, attention_dropout=0.0, train=None)
            with tf.variable_scope("attention_over_to_under"):
                self.over_to_under = attention(self.feature_under, self.feature_over, self.mask_under, self.mask_over,
                                               hidden_size=256, num_heads=2, attention_dropout=0.0, train=None)
            with tf.variable_scope("attention_under_to_over"):
                self.under_to_over = attention(self.feature_over, self.feature_under, self.mask_over, self.mask_under,
                                               hidden_size=256, num_heads=2, attention_dropout=0.0, train=None)

            self.feature_over_2 = tf.concat([self.feature_over, self.whole_to_over, self.under_to_over], axis=2)
            self.feature_under_2 = tf.concat([self.feature_under, self.whole_to_under, self.over_to_under], axis=2)
            self.feature_over_2 = tf.contrib.layers.layer_norm(self.feature_over_2, begin_norm_axis=-1)
            self.feature_under_2 = tf.contrib.layers.layer_norm(self.feature_under_2, begin_norm_axis=-1)

            over_position_encoding = self.position_encoding(self.over_seg_pos, self.over_seg_length, self.mask_over)
            under_position_encoding = self.position_encoding(self.under_seg_pos, self.under_seg_length, self.mask_under)
            whole_pos = tf.linspace(0.0, 1.0, 64)
            whole_pos = tf.reshape(whole_pos, [1, 64, 1])
            whole_pos = tf.tile(whole_pos, [tf.shape(self.input)[0],1,1])
            whole_length = tf.ones_like(self.over_seg_length)*63
            whole_position_encoding = self.position_encoding(whole_pos, whole_length, self.mask_whole)

            fc1 = tf.layers.dense(self.feature_over_2, 512, name='over_fc1', activation=tf.nn.relu, use_bias=False)
            fc1 = tf.contrib.layers.layer_norm(fc1, begin_norm_axis=-1)
            fc2 = tf.layers.dense(fc1, 256, name='over_fc2', activation=None, use_bias=False)
            fc2 = fc2 + tf.layers.dense(over_position_encoding, 256, name='over_position_fc', activation=None, use_bias=False)
            fc2 = tf.nn.relu(fc2)
            fc2 = tf.contrib.layers.layer_norm(fc2, begin_norm_axis=-1)
            self.feature_over_3 = fc2

            fc1 = tf.layers.dense(self.feature_under_2, 512, name='under_fc1', activation=tf.nn.relu, use_bias=False)
            fc1 = tf.contrib.layers.layer_norm(fc1, begin_norm_axis=-1)
            fc2 = tf.layers.dense(fc1, 256, name='under_fc2', activation=None, use_bias=False)
            fc2 = fc2 + tf.layers.dense(under_position_encoding, 256, name='under_position_fc', activation=None, use_bias=False)
            fc2 = tf.nn.relu(fc2)
            fc2 = tf.contrib.layers.layer_norm(fc2, begin_norm_axis=-1)
            self.feature_under_3 = fc2

            fc1 = tf.layers.dense(self.feature_whole, 512, name='whole_fc1', activation=tf.nn.relu, use_bias=False)
            fc1 = tf.contrib.layers.layer_norm(fc1, begin_norm_axis=-1)
            fc2 = tf.layers.dense(fc1, 256, name='whole_fc2', activation=None, use_bias=False)
            fc2 = fc2 + tf.layers.dense(whole_position_encoding, 256, name='whole_position_fc', activation=None, use_bias=False)
            fc2 = tf.nn.relu(fc2)
            fc2 = tf.contrib.layers.layer_norm(fc2, begin_norm_axis=-1)
            self.feature_whole_3 = fc2

            # attention back to whole
            with tf.variable_scope("attention_over_to_whole"):
                self.over_to_whole = attention(self.feature_whole_3, self.feature_over_3, self.mask_whole, self.mask_over,
                                               hidden_size=256, num_heads=2, attention_dropout=0.0, train=None)
            with tf.variable_scope("attention_under_to_whole"):
                self.under_to_whole = attention(self.feature_whole_3, self.feature_under_3, self.mask_whole, self.mask_under,
                                               hidden_size=256, num_heads=2, attention_dropout=0.0, train=None)

            self.feature_whole_4 = tf.concat([self.feature_whole_3, self.over_to_whole, self.under_to_whole], axis=2)
            self.feature_whole_4 = tf.contrib.layers.layer_norm(self.feature_whole_4, begin_norm_axis=-1)

            fc1_a = tf.layers.dense(self.action, 512, name='fc1_a', activation=tf.nn.relu)
            fc1_a = tf.contrib.layers.layer_norm(fc1_a, begin_norm_axis=-1)
            self.state_fc = tf.reduce_max(self.feature_whole_4, axis=-2)
            fc1 = tf.concat([self.state_fc, fc1_a], axis=-1)
            fc2 = tf.layers.dense(fc1, 512, name='fc2', activation=tf.nn.relu)
            fc2 = tf.contrib.layers.layer_norm(fc2, begin_norm_axis=-1)
            fc3 = tf.layers.dense(fc2, 512, name='fc3', activation=tf.nn.relu)
            fc3 = tf.contrib.layers.layer_norm(fc3, begin_norm_axis=-1)
            fc4 = tf.layers.dense(fc3, 512, name='fc4', activation=tf.nn.relu)
            fc4 = tf.contrib.layers.layer_norm(fc4, begin_norm_axis=-1)
            self.q_value = tf.layers.dense(fc4, 1, name='q_value', activation=None)

            # state value
            fc1_v = tf.layers.dense(self.state_fc, 256, name='fc1_v', activation=tf.nn.relu)
            fc1_v = tf.contrib.layers.layer_norm(fc1_v, begin_norm_axis=-1)
            fc2_v = tf.layers.dense(fc1_v, 256, name='fc2_v', activation=tf.nn.relu)
            fc2_v = tf.contrib.layers.layer_norm(fc2_v, begin_norm_axis=-1)
            self.state_value = tf.layers.dense(fc2_v, 1, name='state_value', activation=None)

            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=50)

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
        for k in range(8):
            absolute_pos_sins.append( tf.sin(absolute_pos*k*3.1415) )
        relative_pos = ((absolute_pos - absolute_pos[:,0:1,:]) * 63
                         / tf.reshape(tf.cast(lengths, tf.float32)-1.0, [-1,1,1]))
        relative_pos_sins = []
        for k in range(8):
            relative_pos_sins.append( tf.sin(relative_pos*k*3.1415) )
        position_encoding = tf.concat([absolute_pos] + absolute_pos_sins + [relative_pos] + relative_pos_sins, axis=-1)
        position_encoding = position_encoding * tf.cast(tf.expand_dims(mask, axis=-1), tf.float32)
        return position_encoding


    def make_feed_dict_single(self, obs, over_seg_dict, under_seg_dict):
        feed_dict = {self.input: obs[None],
                     self.over_seg_obs: over_seg_dict['obs'][None],
                     self.over_seg_pos: over_seg_dict['pos'][None],
                     self.over_seg_length: over_seg_dict['length'][None],
                     self.under_seg_obs: under_seg_dict['obs'][None],
                     self.under_seg_pos: under_seg_dict['pos'][None],
                     self.under_seg_length: under_seg_dict['length'][None]}
        return feed_dict

    def make_feed_dict_batch(self, obs, over_seg_dict, under_seg_dict):
        feed_dict = {self.input: obs,
                     self.over_seg_obs: over_seg_dict['obs'],
                     self.over_seg_pos: over_seg_dict['pos'],
                     self.over_seg_length: over_seg_dict['length'],
                     self.under_seg_obs: under_seg_dict['obs'],
                     self.under_seg_pos: under_seg_dict['pos'],
                     self.under_seg_length: under_seg_dict['length']}
        return feed_dict

    def predict_single(self, sess, obs, over_seg_dict, under_seg_dict, action):
        feed_dict = self.make_feed_dict_single(obs, over_seg_dict, under_seg_dict)
        feed_dict[self.action] = action[None]
        q, v = sess.run([self.q_value, self.state_value], feed_dict=feed_dict)
        return q[0], v[0]

    def predict_batch(self, sess, obs, over_seg_dict, under_seg_dict, action):
        feed_dict = self.make_feed_dict_batch(obs, over_seg_dict, under_seg_dict)
        feed_dict[self.action] = action
        q, v = sess.run([self.q_value, self.state_value], feed_dict=feed_dict)
        return q, v

    def predict_single_vf(self, sess, obs, over_seg_dict, under_seg_dict):
        feed_dict = self.make_feed_dict_single(obs, over_seg_dict, under_seg_dict)
        pred, = sess.run([self.state_value], feed_dict=feed_dict)
        return pred[0]

    def predict_batch_vf(self, sess, obs, over_seg_dict, under_seg_dict):
        feed_dict = self.make_feed_dict_batch(obs, over_seg_dict, under_seg_dict)
        pred, = sess.run([self.state_value], feed_dict=feed_dict)
        return pred

    def predict_single_action(self, sess, obs, over_seg_dict, under_seg_dict,
                              init_action_mean=None, init_action_cov=None,
                              iterations=1, q_threshold=0.8):
        CEM_population = 256
        elite_percentage = 0.2
        feed_dict = self.make_feed_dict_single(obs, over_seg_dict, under_seg_dict)
        state_feature = sess.run(self.state_fc, feed_dict=feed_dict)
        mean, cov = init_action_mean, init_action_cov
        for iter in range(iterations):
            action_samples = np.random.multivariate_normal(mean, cov,
                                                           size=CEM_population)
            feed_dict = {self.state_fc:np.tile(state_feature, (CEM_population, 1)),
                         self.action:action_samples}
            qs = sess.run(self.q_value, feed_dict=feed_dict)
            idx = np.argsort(qs[:,0])
            if q_threshold is not None and np.amax(qs) > q_threshold:
                return action_samples[idx[-1]]
            idx = idx[-int(elite_percentage*CEM_population):]
            action_samples = action_samples[idx]
            mean, cov = np.mean(action_samples, axis=0), np.cov(action_samples, rowvar=False)
        return action_samples[-1]

    def predict_batch_action(self, sess, obs, over_seg_dict, under_seg_dict,
                              init_action_mean=None, init_action_cov=None,
                              iterations=1, q_threshold=0.8):
        CEM_population = 256
        elite_percentage = 0.2
        feed_dict = self.make_feed_dict_batch(obs, over_seg_dict, under_seg_dict)
        state_feature = sess.run(self.state_fc, feed_dict=feed_dict)
        actions = []
        for feat, mean, cov in zip(state_feature, init_action_mean, init_action_cov):
            for iter in range(iterations):
                action_samples = np.random.multivariate_normal(mean, cov,
                                                               size=CEM_population)
                feed_dict = {self.state_fc:np.tile(feat, (CEM_population, 1)),
                             self.action:action_samples}
                qs = sess.run(self.q_value, feed_dict=feed_dict)
                idx = np.argsort(qs[:,0])
                max_q = 1 / (1 + np.exp(-np.amax(qs)))
                if q_threshold is not None and max_q > q_threshold:
                    actions.append(action_samples[idx[-1]])
                    break
                idx = idx[-int(elite_percentage*CEM_population):]
                action_samples = action_samples[idx]
                mean, cov = np.mean(action_samples, axis=0), np.cov(action_samples, rowvar=False)
            if q_threshold is None or max_q < q_threshold:
                actions.append(action_samples[idx[-1]])
        return actions

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def setup_optimizer(self, learning_rate):
        with tf.variable_scope(self.scope):

            self.T = 0.2 # 1/temperature
            self.q_gt = tf.placeholder(tf.float32, [None,1])
            self.action_logprob = tf.placeholder(tf.float32, [None,1])

            # Value loss
            self.eval_q_loss = tf.reduce_mean((self.q_gt-tf.sigmoid(self.q_value))**2)
            self.q_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.q_gt, logits=self.q_value))
            v_target = tf.stop_gradient(self.q_value) - self.T * self.action_logprob
            self.v_loss = tf.reduce_mean((tf.squeeze(self.state_value) - v_target)**2)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.q_optimizer = optimizer.minimize(self.q_loss)
            self.v_optimizer = optimizer.minimize(self.v_loss)

            tf.summary.scalar('q_loss', self.q_loss)
            tf.summary.scalar('v_loss', self.v_loss)
            self.merged_summary = tf.summary.merge_all()

    def fit_q(self, sess, obs, over_seg_dict, under_seg_dict, actions, q_targets):
        feed_dict= {self.input:obs,
                    self.over_seg_obs: over_seg_dict['obs'],
                    self.over_seg_pos: over_seg_dict['pos'],
                    self.over_seg_length: over_seg_dict['length'],
                    self.under_seg_obs: under_seg_dict['obs'],
                    self.under_seg_pos: under_seg_dict['pos'],
                    self.under_seg_length: under_seg_dict['length'],
                    self.action:actions,
                    self.q_gt:q_targets
        }
        loss, _ = sess.run([self.eval_q_loss, self.q_optimizer], feed_dict=feed_dict)
        return loss

    def fit_v(self, sess, obs, over_seg_dict, under_seg_dict, actions, action_logprobs):
        feed_dict= {self.input:obs,
                    self.over_seg_obs: over_seg_dict['obs'],
                    self.over_seg_pos: over_seg_dict['pos'],
                    self.over_seg_length: over_seg_dict['length'],
                    self.under_seg_obs: under_seg_dict['obs'],
                    self.under_seg_pos: under_seg_dict['pos'],
                    self.under_seg_length: under_seg_dict['length'],
                    self.action:actions,
                    self.action_logprob:action_logprobs
        }
        loss, _ = sess.run([self.v_loss, self.v_optimizer], feed_dict=feed_dict)
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
    actions = np.random.uniform(size=(4,6))
    qs, vs = model.predict_batch(sess, states, over_seg_dict={'obs':over_seg, 'pos':over_pos/63.0, 'length':over_length},
                                               under_seg_dict={'obs':under_seg, 'pos':under_pos/63.0, 'length':under_length}, action=actions)
    print(qs.shape)
    model.fit_q(sess, states, over_seg_dict={'obs':over_seg, 'pos':over_pos/63.0, 'length':over_length},
                              under_seg_dict={'obs':under_seg, 'pos':under_pos/63.0, 'length':under_length},
                              actions=actions, q_targets=qs)
    model.fit_v(sess, states, over_seg_dict={'obs':over_seg, 'pos':over_pos/63.0, 'length':over_length},
                              under_seg_dict={'obs':under_seg, 'pos':under_pos/63.0, 'length':under_length},
                              actions=actions, action_logprobs=np.zeros((4,1)))

