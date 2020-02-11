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

            over_absolute_pos = self.over_seg_pos
            over_absolute_pos_cos = tf.sin(over_absolute_pos*3.1415)
            over_relative_pos = ((over_absolute_pos - over_absolute_pos[:,0:1,:]) * 63
                                 / tf.reshape(tf.cast(self.over_seg_length, tf.float32)-0.999, [-1,1,1]))
            over_relative_pos_cos = tf.sin(over_relative_pos*3.1415)
            over_position_encoding = tf.concat([over_absolute_pos, over_absolute_pos_cos,
                                               over_relative_pos, over_relative_pos_cos], axis=-1)
            over_position_encoding = over_position_encoding * tf.cast(tf.expand_dims(self.mask_over, axis=-1), tf.float32)
            under_absolute_pos = self.under_seg_pos
            under_absolute_pos_cos = tf.sin(under_absolute_pos*3.1415)
            under_relative_pos = ((under_absolute_pos - under_absolute_pos[:,0:1,:]) * 63
                                   / tf.reshape(tf.cast(self.under_seg_length, tf.float32)-0.999, [-1,1,1]))
            under_relative_pos_cos = tf.sin(under_relative_pos*3.1415)
            under_position_encoding = tf.concat([under_absolute_pos, under_absolute_pos_cos,
                                                under_relative_pos, under_relative_pos_cos], axis=-1)
            under_position_encoding = under_position_encoding * tf.cast(tf.expand_dims(self.mask_under, axis=-1), tf.float32)

            # TODO normalization?
            fc1 = tf.layers.dense(self.feature_over_2, 512, name='over_fc1', activation=tf.nn.relu, use_bias=False)
            fc2 = tf.layers.dense(fc1, 256, name='over_fc2', activation=None, use_bias=False)
            fc2 = fc2 + tf.layers.dense(over_position_encoding, 256, name='over_position_fc', activation=None, use_bias=False)
            fc2 = tf.nn.relu(fc2)
            fc3 = tf.layers.dense(fc2, 1, name='over_fc3', activation=None, use_bias=False)
            fc3_2 = tf.layers.dense(fc2, 1, name='over_fc3_pick', activation=None, use_bias=False)
            fc3, fc3_2 = tf.maximum(fc3, -200), tf.maximum(fc3_2, -200)
            # pad-masked softmax
            fc3 = tf.where_v2(self.mask_over, tf.squeeze(fc3, -1), -1000.0)
            over_summary_weight = tf.nn.softmax(fc3)
            over_summary_feature = tf.reduce_sum(tf.expand_dims(over_summary_weight,-1) * self.feature_over_2, axis=1)

            # pick_logits is the distribution of first action dimension.
            fc3_2 = tf.where_v2(self.mask_over, tf.squeeze(fc3_2, -1), -1000.0)
            self.pick_logits = fc3_2

            fc1 = tf.layers.dense(self.feature_under_2, 512, name='under_fc1', activation=tf.nn.relu, use_bias=False)
            fc2 = tf.layers.dense(fc1, 256, name='under_fc2', activation=None, use_bias=False)
            fc2 = fc2 + tf.layers.dense(under_position_encoding, 256, name='under_position_fc', activation=None, use_bias=False)
            fc2 = tf.nn.relu(fc2)
            fc3 = tf.layers.dense(fc2, 1, name='under_fc3', activation=None, use_bias=False)
            fc3 = tf.maximum(fc3, -200)
            # pad-masked softmax
            fc3 = tf.where_v2(self.mask_under, tf.squeeze(fc3, -1), -1000.0)
            under_summary_weight = tf.nn.softmax(fc3)
            under_summary_feature = tf.reduce_sum(tf.expand_dims(under_summary_weight,-1) * self.feature_under_2, axis=1)

            final_feature = tf.concat([over_summary_feature, under_summary_feature], axis=-1)
            gaussian_mean_init = tf.constant_initializer([0.0,0.0,0.0,0.0,0.1])
            self.gaussian_mean = self.dense(final_feature, 'gaussian_mean', action_dim-1, activation=None,
                                            scale=0.01, bias_init=gaussian_mean_init)
            self.gaussian_std = self.dense(final_feature, 'gaussian_std', action_dim-1, activation=tf.nn.softplus,
                                            scale=0.01, bias_init=tf.constant_initializer([-1.5,-1.5,-1.5,-1.5,-2.0]))
            self.gaussian = tfp.distributions.MultivariateNormalDiag(loc=self.gaussian_mean,
                                                                     scale_diag = self.gaussian_std+0.001)

            self.categorical = tfp.distributions.Categorical(logits=self.pick_logits) # have to use prob instead of logits
            sample_node = self.categorical.sample()
            sample_node = tf.expand_dims(sample_node, -1) # to use gather
            self.sample_action_first = tf.gather(self.over_seg_pos, sample_node, batch_dims=1, axis=1)
            self.sample_action_first = tf.squeeze(self.sample_action_first, axis=1)
            self.sample_action_second = self.gaussian.sample()
            self.sample_action = tf.concat([self.sample_action_first, self.sample_action_second], axis=-1)

            ML_node = tf.argmax(self.pick_logits, axis=-1) # maximum likelyhood
            ML_node = tf.expand_dims(ML_node, -1)
            self.ML_action_first = tf.gather(self.over_seg_pos, ML_node, batch_dims=1, axis=1)
            self.ML_action_first = tf.squeeze(self.ML_action_first, axis=1)
            self.ML_action = tf.concat([self.ML_action_first, self.gaussian_mean], axis=-1)

            # state value
            self.state_value = self.dense(final_feature, 'state_value', 1, activation=None)

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

    def predict_single(self, sess, obs, over_seg_dict, under_seg_dict, explore=False):
        feed_dict = {self.input: obs[None],
                     self.over_seg_obs: over_seg_dict['obs'][None],
                     self.over_seg_pos: over_seg_dict['pos'][None],
                     self.over_seg_length: over_seg_dict['length'][None],
                     self.under_seg_obs: under_seg_dict['obs'][None],
                     self.under_seg_pos: under_seg_dict['pos'][None],
                     self.under_seg_length: under_seg_dict['length'][None]}
        if explore:
            pred, = sess.run([self.sample_action], feed_dict=feed_dict)
        else:
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
            pred, = sess.run([self.sample_action], feed_dict=feed_dict)
        else:
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
            self.action = tf.placeholder(tf.float32, self.sample_action.shape)
            self.advantage = tf.placeholder(tf.float32, [None])
            self.reward = tf.placeholder(tf.float32, [None])

            # Policy loss
            self.action_first, self.action_second = tf.split(self.action, [1,5], axis=-1)
            node_category = tf.argmin(tf.abs(self.action_first-self.over_seg_pos[:,:,0]), axis=-1)
            neglogpac = -self.categorical.log_prob(node_category) - self.gaussian.log_prob(self.action_second)
            neglogpac = tf.where(tf.reduce_min(tf.abs(self.action_first-self.over_seg_pos[:,:,0]), axis=-1) < 1/63.0,
                                 neglogpac, tf.zeros_like(neglogpac))
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
        feed_dict= {self.input:obs,
                    self.action:actions,
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
    model.setup_optimizer(1e-3, 1e-3, 1e-3)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    states = np.zeros((4,64,3))
    states[:,:,0]=np.linspace(0,1,64)
    over_seg=states[:,:32,:]
    over_pos=np.ones((4,32,1))
    over_length=np.ones((4,))*32
    over_length=over_length.astype(np.int32)
    under_seg=states[:,32:60,:]
    under_pos=np.ones((4,28,1))
    under_length=np.ones((4,))*28
    under_length=under_length.astype(np.int32)
    action=sess.run(model.sample_action, feed_dict={model.input:states,
                                                    model.over_seg_obs:over_seg, model.over_seg_pos:over_pos, model.over_seg_length:over_length,
                                                    model.under_seg_obs:under_seg, model.under_seg_pos:under_pos, model.under_seg_length:under_length,
                                                    })
    print(action.shape)

    _ =sess.run(model.optimizer,  feed_dict={model.input:states,
                                             model.over_seg_obs:over_seg, model.over_seg_pos:over_pos, model.over_seg_length:over_length,
                                             model.under_seg_obs:under_seg, model.under_seg_pos:under_pos, model.under_seg_length:under_length,
                                             model.action:action, model.advantage:np.ones((4,)), model.reward:np.ones((4,))
                                             })

