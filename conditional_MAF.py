import tensorflow as tf
import tensorflow_probability as tfp
import pdb

def masked_autoregressive_conditional(condition_feature,
                                      hidden_layers,
                                      shift_only=False,
                                      activation=tf.nn.relu,
                                      log_scale_min_clip=-5.,
                                      log_scale_max_clip=3.,
                                      log_scale_clip_gradient=False,
                                      name=None,
                                      *args,  # pylint: disable=keyword-arg-before-vararg
                                      **kwargs):
  """Build the Masked Autoregressive Density Estimator (Germain et al., 2015).
  This will be wrapped in a make_template to ensure the variables are only
  created once. It takes the input and returns the `loc` ('mu' in [Germain et
  al. (2015)][1]) and `log_scale` ('alpha' in [Germain et al. (2015)][1]) from
   the MADE network.
  Warning: This function uses `masked_dense` to create randomly initialized
  `tf.Variables`. It is presumed that these will be fit, just as you would any
  other neural architecture which uses `tf.layers.dense`.
  """

  name = name or "masked_autoregressive_default_template"
  with tf.compat.v2.name_scope(name):

    def _fn(x):
      """MADE parameterized via `masked_autoregressive_default_template`."""
      # TODO(b/67594795): Better support of dynamic shape.
      input_depth = x.get_shape().as_list()[-1]
      if input_depth is None:
        raise NotImplementedError(
            "Rightmost dimension must be known prior to graph execution.")
      input_shape = tf.shape(input=x)
      output_units = (1 if shift_only else 2) * input_depth
      for i, units in enumerate(hidden_layers+[output_units]):
        x = tfp.bijectors.masked_dense(
            inputs=x,
            units=units,
            num_blocks=input_depth,
            exclusive=True if i == 0 else False,
            activation=None,
            *args,  # pylint: disable=keyword-arg-before-vararg
            **kwargs)
        if i==0:
          x = x + tf.layers.dense(condition_feature, units, use_bias=False)
        if i < len(hidden_layers) and activation is not None:
          x = activation(x)
      if shift_only:
        x = tf.reshape(x, shape=input_shape)
        return x, None
      x = tf.reshape(x, shape=tf.concat([input_shape, [2]], axis=0))
      shift, log_scale = tf.unstack(x, num=2, axis=-1)
      which_clip = (
          tf.clip_by_value
          if log_scale_clip_gradient else tfp.math.clip_by_value_preserve_gradient)
      log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
      return shift, log_scale

    return tf.compat.v1.make_template(name, _fn)


if __name__ == "__main__":
    conditional_feat = tf.ones((6,32))
    func = masked_autoregressive_conditional(conditional_feat, [])
    y = tf.constant([[0,0,0,0,0],
                     [1,0,0,0,0],
                     [1,1,0,0,0],
                     [1,1,1,0,0],
                     [1,1,1,1,0],
                     [1,1,1,1,1]], dtype=tf.float32)
    shift, log_scale = func(y)

    reference_func = tfp.bijectors.masked_autoregressive_default_template(hidden_layers=[5], activation=None)
    ref_shift, ref_log_scale = reference_func(y)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    val_shift, val_scale = sess.run([shift, log_scale])
    pdb.set_trace()
