# modified from the official model repo

import tensorflow as tf

def split_heads(x, num_heads):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, hidden_size]
    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        depth = x.shape[2] // num_heads
        x = tf.reshape(x, [batch_size, length, num_heads, depth])
        return tf.transpose(x, [0, 2, 1, 3])

def combine_heads(x):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[2]
        hidden_size = x.shape[1]*x.shape[3]
        x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
        return tf.reshape(x, [batch_size, length, hidden_size])

def attention(x, y, mask_x, mask_y, hidden_size, num_heads, attention_dropout, train):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, D1]
      y: a tensor with shape [batch_size, length_y, D2]
      mask_x / mask_y: a tensor with shape [batch_size, length_x]
      hidden_size: total hidden size, will be divided by num_heads
    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    assert(hidden_size % num_heads == 0)
    q = tf.layers.dense(x, hidden_size, use_bias=False, name='q')
    k = tf.layers.dense(y, hidden_size, use_bias=False, name='k')
    v = tf.layers.dense(y, hidden_size, use_bias=False, name='v')

    # Split q, k, v into heads.
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (hidden_size // num_heads)
    q *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.matmul(q, k, transpose_b=True)
    # mask weights where padded
    mask_x = tf.expand_dims(mask_x, -1)
    mask_y = tf.expand_dims(mask_y, -2)
    mask_x = tf.expand_dims(mask_x, 1)
    mask_y = tf.expand_dims(mask_y, 1)
    logits = tf.where_v2(mask_y, logits, -1000.0)
    weights = tf.nn.softmax(logits, name="attention_weights")
    weights = weights * tf.cast(mask_x, tf.float32)
    if attention_dropout > 0:
      weights = tf.layers.dropout(weights, attention_dropout, training=train)
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = combine_heads(attention_output)
    return attention_output
