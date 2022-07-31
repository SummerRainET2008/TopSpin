#coding: utf8
#author: Tian Xia

from palframe.tf_2x import *


class PrePostProcessingWrapper:
  def __init__(self, layer, layer_postprocess_dropout):
    self._layer = layer
    self._layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self._dropout = tf.keras.layers.Dropout(layer_postprocess_dropout)

  def __call__(self, x, *args, **kwargs):
    y = self._layer(self._layer_norm(x), *args, **kwargs)
    y = self._dropout(y, kwargs["training"])

    return x + y


class FFN(tf.keras.layers.Layer):
  def __init__(self, ffn_dim, hidden_size, relu_dropout):
    super(FFN, self).__init__()

    self._dropout = tf.keras.layers.Dropout(relu_dropout)
    self._filter_dense_layer = tf.keras.layers.Dense(ffn_dim,
                                                     activation=tf.nn.relu,
                                                     name="filter_layer")
    self._output_dense_layer = tf.keras.layers.Dense(hidden_size,
                                                     name="output_layer")

  def call(self, x, *args, **kwargs):
    '''
      x: tensor with shape [batch_size, length, hidden_size]
      Returns: [batch_size, length, hidden_size]
    '''
    output = self._filter_dense_layer(x)
    output = self._dropout(output, kwargs["training"])
    output = self._output_dense_layer(output)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, hidden_size, num_heads, atten_dropout):
    assert hidden_size % num_heads == 0

    super(MultiHeadAttention, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._atten_dropout = tf.keras.layers.Dropout(atten_dropout)

    # Layers for linearly projecting the queries, keys, and values.
    self._q_dense_layer = tf.keras.layers.Dense(hidden_size,
                                                use_bias=False,
                                                name="q")
    self._k_dense_layer = tf.keras.layers.Dense(hidden_size,
                                                use_bias=False,
                                                name="k")
    self._v_dense_layer = tf.keras.layers.Dense(hidden_size,
                                                use_bias=False,
                                                name="v")

    self._output_layer = tf.keras.layers.Dense(hidden_size,
                                               use_bias=False,
                                               name="output_transform")

  def _split_heads(self, x):
    '''
      x:      [batch_size, length, hidden_size]
      return: [batch_size, num_heads, length, hidden_size/num_heads]
    '''
    length = x.shape[1]
    depth = (self._hidden_size // self._num_heads)
    x = tf.reshape(x, [-1, length, self._num_heads, depth])

    return tf.transpose(x, [0, 2, 1, 3])

  def _combine_heads(self, x):
    '''
      x:      [batch_size, num_heads, length, hidden_size/num_heads]
      return: [batch_size, length, hidden_size]
    '''
    length = x.shape[2]
    x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]

    return tf.reshape(x, [-1, length, self._hidden_size])

  def _gen_padding_mask(self, seq_len, max_seq_len):
    padding_mask = tf.sequence_mask(seq_len,
                                    maxlen=max_seq_len,
                                    dtype=tf.float32)
    padding_mask = (1 - padding_mask) * -1e10
    padding_mask = tf.expand_dims(tf.expand_dims(padding_mask, 1), 1)

    return padding_mask

  def call(self, query, key_values, *args, **kwargs):
    '''
      query:      [batch_size, length_x, hidden_size]
      key_values: [batch_size, length_y, hidden_size]
      bias: padding mask.
    Returns: [batch_size, length_x, hidden_size]
    '''
    q = self._q_dense_layer(query)
    k = self._k_dense_layer(key_values)
    v = self._v_dense_layer(key_values)

    q = self._split_heads(q)
    k = self._split_heads(k)
    v = self._split_heads(v)

    depth = (self._hidden_size // self._num_heads)
    q *= depth**-0.5

    logits = tf.matmul(q, k, transpose_b=True)

    # top_values, top_indices = tf.math.top_k(logits, k=8)
    # feat_num = query.shape[1]
    # mask = nlp_tf.to_double(
    #   1 - tf.reduce_sum(tf.one_hot(top_indices, feat_num), 3)
    # ) * -1e10

    assert "seq_len" in kwargs
    padding_mask = self._gen_padding_mask(kwargs["seq_len"],
                                          key_values.shape[1])
    logits += padding_mask
    weights = tf.nn.softmax(logits)

    assert "training" in kwargs
    weights = self._atten_dropout(weights, kwargs["training"])

    attention_output = weights @ v

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self._combine_heads(attention_output)
    attention_output = self._output_layer(attention_output)

    return attention_output


class SelfMultiHeadAttention(MultiHeadAttention):
  def call(self, query, *args, **kwargs):
    # kwargs["key_values"] = query
    return super(SelfMultiHeadAttention, self).call(query, query, *args,
                                                    **kwargs)


class EncoderStack(tf.keras.layers.Layer):
  def __init__(self, layer_num, hidden_size, num_heads, ffn_dim, relu_dropout,
               atten_dropout, layer_postprocess_dropout):
    super(EncoderStack, self).__init__()
    self._all_layer_blocks = []
    for _ in range(layer_num):
      self_attention_layer = SelfMultiHeadAttention(hidden_size, num_heads,
                                                    atten_dropout)
      ffn = FFN(ffn_dim, hidden_size, relu_dropout)

      self._all_layer_blocks.append([
          PrePostProcessingWrapper(self_attention_layer,
                                   layer_postprocess_dropout),
          PrePostProcessingWrapper(ffn, layer_postprocess_dropout)
      ])

    self._output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)

  def call(self, encoder_inputs, *args, **kwargs):
    '''Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer.
        [batch_size, 1, 1, input_length]
      inputs_padding: P

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    '''

    for n, layer in enumerate(self._all_layer_blocks):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      encoder_inputs = self_attention_layer(encoder_inputs, *args, **kwargs)
      encoder_inputs = feed_forward_network(encoder_inputs, *args, **kwargs)

    return self._output_normalization(encoder_inputs)


class Encoder(tf.keras.layers.Layer):
  def __init__(self, layer_num, hidden_size, num_heads, ffn_dim, relu_dropout,
               atten_dropout, layer_postprocess_dropout):
    super(Encoder, self).__init__()
    self._input_dense = tf.keras.layers.Dense(hidden_size)
    self._encoder_stack = EncoderStack(
        layer_num=layer_num,
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        relu_dropout=relu_dropout,
        atten_dropout=atten_dropout,
        layer_postprocess_dropout=layer_postprocess_dropout,
    )
    self._input_dense = tf.keras.layers.Dense(hidden_size,
                                              activation=tf.nn.leaky_relu)

  def call(self, input_x: tf.Tensor, seq_len: tf.Tensor, *args, **kwargs):
    '''
    input_x: [batch, seq_len, feat_num]
    input_length: [batch]
    '''
    assert "training" in kwargs
    kwargs["seq_len"] = seq_len
    x = self._input_dense(input_x)
    x = self._encoder_stack(x, *args, **kwargs)

    return x
