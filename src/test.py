import tensorflow as tf
import numpy as np

# n_amino_acids = 700
# n_features = 44
# n_out_classes = 4
#
# # Weight & Bias of the model
# weights = {
#     # 11x11 conv, 1 input, 32 outputs
#     'wc1': tf.get_variable(name='wc1', shape=[1, 3, 1, 2], dtype=tf.float32,
#                            initializer=tf.contrib.layers.xavier_initializer_conv2d()),
#     # 6x6 conv, 32 inputs, 64 outputs
#     'wc2': tf.get_variable(name='wc2', shape=[1, 3, 2, 4], dtype=tf.float32,
#                            initializer=tf.contrib.layers.xavier_initializer_conv2d()),
#
#     'wc3': tf.get_variable(name='wc3', shape=[1, 3, 4, 1], dtype=tf.float32,
#                            initializer=tf.contrib.layers.xavier_initializer_conv2d()),
# }
# biases = {
#     'bc1': tf.get_variable(name='bc1', shape=[2], dtype=tf.float32,
#                            initializer=tf.contrib.layers.xavier_initializer()),
#     'bc2': tf.get_variable(name='bc2', shape=[4], dtype=tf.float32,
#                            initializer=tf.contrib.layers.xavier_initializer()),
#     'bc3': tf.get_variable(name='bc3', shape=[1], dtype=tf.float32,
#                            initializer=tf.contrib.layers.xavier_initializer())
# }
#
# weights2 = {
#     # 11x11 conv, 1 input, 32 outputs
#     'wc1': tf.random_normal(name='wc12', shape=[1, 3, 1, 2], dtype=tf.float32),
#     # 6x6 conv, 32 inputs, 64 outputs
#     'wc2': tf.random_normal(name='wc22', shape=[1, 3, 2, 4], dtype=tf.float32),
#
#     'wc3': tf.random_normal(name='wc32', shape=[1, 3, 4, 1], dtype=tf.float32)
# }
# biases2 = {
#     'bc1': tf.get_variable(name='bc12', shape=[2], dtype=tf.float32,
#                            initializer=tf.contrib.layers.xavier_initializer()),
#     'bc2': tf.get_variable(name='bc22', shape=[4], dtype=tf.float32,
#                            initializer=tf.contrib.layers.xavier_initializer()),
#     'bc3': tf.get_variable(name='bc32', shape=[1], dtype=tf.float32,
#                            initializer=tf.contrib.layers.xavier_initializer())
# }
#
# # Convolution Layer 1
# var = tf.random_normal(shape=[1, 24, 9, 1], dtype=tf.float32)
#
# conv1_layer1 = tf.nn.conv2d(input=var, filter=weights2['wc1'],
#                             strides=[1, 1, 1, 1], padding='VALID')
# conv1_layer2 = tf.nn.bias_add(conv1_layer1, biases['bc1'])
# conv1_layer3 = tf.nn.relu(conv1_layer2)
# #
# # X = tf.placeholder(shape=[1, 2, 2, 1], dtype=tf.float32)
#
# max1_layer = tf.nn.max_pool(conv1_layer3, ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
# #
# # Convolution Layer 2
# conv2_layer = tf.nn.conv2d(input=max1_layer, filter=weights2['wc2'], strides=[1, 1, 1, 1], padding='VALID')
# conv2_layer = tf.nn.bias_add(conv2_layer, biases['bc2'])
# conv2_layer = tf.nn.relu(conv2_layer)
# #
# # # Max Pooling 2
# max2_layer = tf.nn.max_pool(conv2_layer, ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
# #
# # # Fully connected layer
# # # Reshape conv2 output to fit fully connected layer input
# conv3_layer = tf.nn.conv2d(input=max2_layer, filter=weights2['wc3'], strides=[1, 1, 1, 1], padding='VALID')
# conv3_layer = tf.nn.bias_add(conv3_layer, biases['bc3'])
# conv3_layer = tf.nn.relu(conv3_layer)
# #
# conv3_layer = tf.reshape(conv3_layer, shape=[-1, 24, 1])
# # Output, regression prediction
#
x1 = tf.constant(1., shape=[4, 3, 2])
x2 = tf.constant(1., shape=[4, 2, 1])

c1 = tf.constant(value=[[[1, 3, 12, 2],
                         [10, 11, 11, 12],
                         [90, 88, 85, 12],
                         [86, 12, 123, 23]],
                        [[1, 1, 2, 1],
                        [2, 2, 3, 2],
                        [14, 15, 14, 13],
                        [192, 192, 195, 192]]], dtype=tf.float32)
#
# c1 = tf.constant(value=[[[0, 0, 1, 0],
#                         [0, 0, 1, 0],
#                         [0, 1, 0, 0],
#                         [0, 0, 1, 0]]], dtype=tf.float32)
#
# c1 = tf.constant(value=[0, 0, 3, 0], dtype=tf.float32)
# c2 = tf.constant(value=[0, 0, 3, 0], dtype=tf.float32)


#
c2 = tf.constant(value=[[[0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0]],
                        [[0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]]], dtype=tf.float32)



# c1 = tf.nn.softmax(c1)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=c1, labels=c2)
mean_loss = tf.reduce_mean(loss)


max_indices_c1 = tf.arg_max(c1, dimension=2)
max_indices_c2 = tf.arg_max(c2, dimension=2)

one_hot = tf.one_hot(max_indices_c1, 4)

prod = tf.matmul(x1, x2)

acc = tf.equal(tf.arg_max(c1, dimension=2), tf.arg_max(c2, dimension=2))
tf_acc = tf.contrib.metrics.accuracy(tf.arg_max(c2, dimension=2), tf.arg_max(c1, dimension=2))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(max_indices_c1))
    print(sess.run(max_indices_c2))
    print("One hot:")
    print(sess.run(one_hot))
    print("Y:")
    print(sess.run(c2))
    print("Accuracy:")
    print(sess.run(acc))
    print(sess.run(tf_acc))