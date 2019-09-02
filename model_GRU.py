import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pdb

class Model:
    def __init__(self, reward_key):
        self.scope=reward_key

    def build(self, input=None, action_dim=6, action_init=None):

        with tf.variable_scope(self.scope):
            if input is not None:
                self.input = input
            else:
                self.input = tf.placeholder(dtype=tf.float32, shape=[None,64,3])

            #cell = tf.nn.rnn_cell.GRUCell(512, activation=tf.nn.relu6, name='gru_cell')
            #self.biLSTM, self.biLSTM_final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell, self.input,
            #                                                                       dtype = tf.float32, time_major=False)

            # self.biLSTM stores (hidden_fw, hidden_bw)
            # self.feature = tf.concat([self.biLSTM[0], self.biLSTM[1], self.input], axis=2)

            # generating attention weight matrix
            # TODO: attention is all you need?
            # self.attention_feature = tf.matmul(attention_w, self.feature)
            # self.combined_feature = tf.concat([self.feature, self.attention_feature], axis=2)

            #self.feature = tf.concat([self.biLSTM_final_state[0], self.biLSTM_final_state[1]], axis=-1)
            #fc1 = self.dense(self.feature, 'fc1', 512, 'relu')
            #fc2 = self.dense(fc1, 'fc2', 256, 'relu')
            gaussian_mean_init = tf.constant_initializer(action_init) if action_init is not None else tf.zeros_initializer()
            input = tf.layers.flatten(self.input)
            self.gaussian_mean = self.dense(input, 'gaussian_mean', action_dim, activation=None,
                                            scale=0.01, bias_init=gaussian_mean_init)
            self.gaussian_logstd = tf.get_variable('gaussian_logstd', shape=[1,action_dim],
                                            initializer = tf.constant_initializer([-2.0,-1.5,-1.5,-1.5,-1.5,-2.0]))
            self.gaussian = tfp.distributions.MultivariateNormalDiag(loc=self.gaussian_mean,
                                                                     scale_diag = tf.exp(self.gaussian_mean*0.0+self.gaussian_logstd)) # Bx6
            self.sample_action = self.gaussian.sample()

            self.saver = tf.train.Saver(var_list=self.get_trainable_variables(), max_to_keep=1000000)

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

    def predict_single(self, sess, input, explore=False):
        if explore:
            pred, = sess.run([self.sample_action], feed_dict={self.input:input[None]})
        else:
            pred, = sess.run([self.gaussian_mean], feed_dict={self.input:input[None]})
        return pred[0]

    def predict_batch(self, sess, inputs, explore=False):
        if explore:
            pred, = sess.run([self.sample_action], feed_dict={self.input:inputs})
        else:
            pred, = sess.run([self.gaussian_mean], feed_dict={self.input:inputs})
        return pred

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def setup_optimizer(self, learning_rate, ent_coef, max_grad_norm):
        with tf.variable_scope(self.scope):
            self.action = tf.placeholder(tf.float32, self.sample_action.shape)
            self.advantage = tf.placeholder(tf.float32, [None])
            self.reward = tf.placeholder(tf.float32, [None])

            # Policy loss
            neglogpac = -self.gaussian.log_prob(self.action)
            self.pg_loss = tf.reduce_mean(self.advantage * neglogpac)
            # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
            self.entropy = tf.reduce_mean(self.gaussian.entropy())
            # Value loss
            # self.vf_loss = tf.nn.l2_loss(tf.squeeze(self.vf), self.reward) / self.reward.shape[0]
            self.loss = self.pg_loss - self.entropy*ent_coef # + vf_loss * vf_coef

            params = self.get_trainable_variables()
            grads = tf.gradients(self.loss, params)
            if max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.optimizer = optimizer.apply_gradients(grads)

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('pg_loss', self.pg_loss)
            #tf.summary.scalar('vf_loss', self.vf_loss)
            tf.summary.scalar('entropy', self.entropy)
            self.merged_summary = tf.summary.merge_all()

    def fit(self, sess, inputs, actions, advantages, rewards):
        _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.input:inputs,
                                                                   self.action:actions,
                                                                   self.advantage:advantages,
                                                                   self.reward:rewards})
        return loss

    def save(self, sess, file_dir, step):
        self.saver.save(sess, file_dir, global_step=step)

    def load(self, sess, snapshot):
        self.saver.restore(sess, snapshot)

