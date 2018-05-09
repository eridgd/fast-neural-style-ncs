import tensorflow as tf

WEIGHTS_INIT_STDEV = .1

def net(image):
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 127.5 + 127.5
    return preds

def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    net = tf.layers.conv2d(net, filters=num_filters, kernel_size=filter_size, strides=strides, padding='SAME', kernel_initializer=initializer)

    net = tf.layers.batch_normalization(net, fused=True)
    if relu:
        net = tf.nn.relu(net)

    return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    net = tf.layers.conv2d_transpose(net, filters=num_filters, kernel_size=filter_size, strides=strides, padding='SAME', kernel_initializer=initializer)

    net = tf.layers.batch_normalization(net, fused=True)
    return tf.nn.relu(net)

def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)

# def reduce_var(x, axis=None, keepdims=False):
#     """Variance of a tensor, alongside the specified axis."""
#     m = tf.reduce_mean(x, axis=axis, keep_dims=True)
#     devs_squared = tf.square(x - m)
#     return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

# def reduce_std(x, axis=None, keepdims=False):
#     """Standard deviation of a tensor, alongside the specified axis."""
#     return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

# def _instance_norm(net, train=True):
#     batch, rows, cols, channels = [i.value for i in net.get_shape()]
#     var_shape = [channels]

#     mu = tf.reduce_mean(net, axis=[1,2], keep_dims=True)
#     sigma = reduce_std(net, axis=[1,2], keepdims=True)

#     # mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
#     shift = tf.Variable(tf.zeros(var_shape))
#     scale = tf.Variable(tf.ones(var_shape))
#     # epsilon = 1e-3
#     # normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
#     normalized = (net-mu) / sigma
#     return scale * normalized + shift

# def _conv_init_vars(net, out_channels, filter_size, transpose=False):
#     _, rows, cols, in_channels = [i.value for i in net.get_shape()]
#     if not transpose:
#         weights_shape = [filter_size, filter_size, in_channels, out_channels]
#     else:
#         weights_shape = [filter_size, filter_size, out_channels, in_channels]

#     # initializer = tf.contrib.layers.xavier_initializer_conv2d()
#     # weights_init = tf.Variable(initializer(shape=weights_shape), dtype=tf.float32)

#     weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
#     return weights_init
