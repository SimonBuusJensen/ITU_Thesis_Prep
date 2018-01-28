import numpy as np
import tensorflow as tf


class CNNModel:
    def __init__(self):
        print("Building model...")
        self.n_amino_acids = 700
        self.n_features = 44
        self.n_out_classes = 4

        # Weight & Bias of the model
        self.weights = {
            # 11x11 conv, 1 input, 32 outputs
            'wc1': tf.get_variable(name='wc1', shape=[11, 44, 100], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer()),
            # 6x6 conv, 32 inputs, 64 outputs
            'wc2': tf.get_variable(name='wc2', shape=[6, 100, 50], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer()),

            'wc3': tf.get_variable(name='wc3', shape=[6, 50, 4], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer()),
        }
        self.biases = {
            'bc1': tf.get_variable(name='bc1', shape=[100], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable(name='bc2', shape=[50], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable(name='bc3', shape=[4], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())

        }

        # self.weights = {
        #     # 11x11 conv, 1 input, 32 outputs
        #     'wc1': tf.get_variable(name='wc1', shape=[1, 11, 1, 60], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer()),
        #     'wc2': tf.get_variable(name='wc2', shape=[1, 11, 60, 30], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer()),
        #
        #     'wc3': tf.get_variable(name='wc3', shape=[1, 11, 30, 15], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer()),
        #     'wc4': tf.get_variable(name='wc4', shape=[1, 11, 15, 1], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer())
        # }
        # self.biases = {
        #     'bc1': tf.get_variable(name='bc1', shape=[60], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer()),
        #     'bc2': tf.get_variable(name='bc2', shape=[30], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer()),
        #     'bc3': tf.get_variable(name='bc3', shape=[15], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer()),
        #     'bc4': tf.get_variable(name='bc4', shape=[1], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer())
        # }

        # self.weights = {
        #     # 11x11 conv, 1 input, 32 outputs
        #     'wc1': tf.Variable(initial_value=tf.random_normal(name='wc1', shape=[11, 44, 44], dtype=tf.float32,
        #                                                       mean=0, stddev=1)),
        #     # 6x6 conv, 32 inputs, 64 outputs
        #     'wc2': tf.Variable(initial_value=tf.random_normal(name='wc2', shape=[11, 44, 44], dtype=tf.float32,
        #                                                       mean=0, stddev=1)),
        #
        #     'wc3': tf.Variable(initial_value=tf.random_normal(name='wc3', shape=[11, 44, 4], dtype=tf.float32,
        #                                                       mean=0, stddev=1))
        # }
        # self.biases = {
        #     'bc1': tf.get_variable(name='bc1', shape=[88], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer()),
        #     'bc2': tf.get_variable(name='bc2', shape=[44], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer()),
        #     'bc3': tf.get_variable(name='bc3', shape=[4], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer())
        #
        # }
        self.build_model()

    def build_model(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_amino_acids, self.n_features])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_amino_acids, self.n_out_classes])

        self.logits = self.cnn_model_conv1(self.X)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

    # def cnn_model(self, x):
    #     # Convolution Layer 1
    #     conv1_layer = tf.nn.conv2d(input=x, filter=self.weights['wc1'], strides=[1, 1, 1, 1], padding='VALID')
    #     conv1_layer = tf.nn.bias_add(conv1_layer, self.biases['bc1'])
    #     conv1_layer = tf.nn.relu(conv1_layer)
    #
    #     # max1_layer = tf.nn.max_pool(conv1_layer, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
    #
    #     # Convolution Layer 2
    #     conv2_layer = tf.nn.conv2d(input=conv1_layer, filter=self.weights['wc2'], strides=[1, 1, 1, 1], padding='VALID')
    #     conv2_layer = tf.nn.bias_add(conv2_layer, self.biases['bc2'])
    #     conv2_layer = tf.nn.relu(conv2_layer)
    #
    #     # Max Pooling 2
    #     # max2_layer = tf.nn.max_pool(conv2_layer, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
    #
    #     conv3_layer = tf.nn.conv2d(input=conv2_layer, filter=self.weights['wc3'], strides=[1, 1, 1, 1], padding='VALID')
    #     conv3_layer = tf.nn.bias_add(conv3_layer, self.biases['bc3'])
    #     conv3_layer = tf.nn.relu(conv3_layer)
    #
    #     conv4_layer = tf.nn.conv2d(input=conv3_layer, filter=self.weights['wc4'], strides=[1, 1, 1, 1], padding='VALID')
    #     conv4_layer = tf.nn.bias_add(conv4_layer, self.biases['bc4'])
    #
    #     conv4_layer = tf.reshape(conv4_layer, shape=[-1, 700, 4])
    #
    #     return conv4_layer

    def cnn_model_conv1(self, x):
        # Convolution Layer 1
        conv1_layer = tf.nn.conv1d(value=x, filters=self.weights['wc1'], stride=1, padding='SAME')
        # conv1_layer = tf.nn.conv2d(input=x, filter=self.weights['wc1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1_layer = tf.nn.bias_add(conv1_layer, self.biases['bc1'])
        conv1_layer = tf.nn.relu(conv1_layer)

        # max1_layer = tf.nn.max_pool(conv1_layer, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')

        # Convolution Layer 2
        conv2_layer = tf.nn.conv1d(value=conv1_layer, filters=self.weights['wc2'], stride=1, padding='SAME')
        # conv2_layer = tf.nn.conv2d(input=conv1_layer, filter=self.weights['wc2'], strides=[1, 1, 1, 1], padding='VALID')
        conv2_layer = tf.nn.bias_add(conv2_layer, self.biases['bc2'])
        conv2_layer = tf.nn.relu(conv2_layer)

        # Max Pooling 2
        # max2_layer = tf.nn.max_pool(conv2_layer, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')

        conv3_layer = tf.nn.conv1d(value=conv2_layer, filters=self.weights['wc3'], stride=1, padding='SAME')
        # conv3_layer = tf.nn.conv2d(input=conv2_layer, filter=self.weights['wc3'], strides=[1, 1, 1, 1], padding='VALID')
        conv3_layer = tf.nn.bias_add(conv3_layer, self.biases['bc3'])
        return conv3_layer

    def train(self, x_train, y_train, x_test, y_test, x_valid, y_valid, train_end_idx, test_end_idx, valid_end_idx):
        optimizer = tf.train.AdamOptimizer(0.002).minimize(self.loss)

        prediction = tf.one_hot(tf.argmax(self.logits, 2), depth=4)
        # tf_acc = tf.contrib.metrics.accuracy(tf.arg_max(prediction, dimension=2), tf.arg_max(self.Y, dimension=2))
        # acc = tf.equal(tf.arg_max(prediction, dimension=2), tf.arg_max(self.Y, dimension=2))

        batch_size = 32
        n_train_batches = int(len(x_train) / batch_size)
        n_test_batches = int(len(x_test) / batch_size)
        n_valid_batches = int(len(x_valid) / batch_size)

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            first_epoch = True
            for epoch in range(50):

                epoch_loss = 0
                print("Shuffling train data...")
                permutation = np.random.permutation(x_train.shape[0])
                x_train = x_train[permutation]
                y_train = y_train[permutation]
                train_end_idx = train_end_idx[permutation]

                first_batch = True
                for i in range(n_train_batches):
                    x_train_batch = x_train[i * batch_size:(i + 1) * batch_size]
                    y_train_batch = y_train[i * batch_size:(i + 1) * batch_size]
                    logits_x = sess.run([self.logits], feed_dict={self.X: x_train_batch})

                    batch_loss, _ = sess.run([self.loss, optimizer],
                                             feed_dict={self.X: x_train_batch, self.Y: y_train_batch})

                    epoch_loss += batch_loss

                    if (i + 1) % 10 == 0 and not first_batch:
                        logits = sess.run([self.logits], feed_dict={self.X: x_train_batch})
                        train_pred = sess.run([prediction], feed_dict={self.X: x_train_batch})

                        # acc1 = sess.run([acc], feed_dict={self.X: x_train_batch, self.Y: y_train_batch})
                        # acc2 = sess.run([tf_acc], feed_dict={self.X: x_train_batch, self.Y: y_train_batch})

                        equal = 0
                        for ii in range(len(train_pred[0])):
                            for jj in range(
                                    len(train_pred[0][ii][:train_end_idx[i * batch_size:(i + 1) * batch_size][ii][0]])):
                                equal += int(np.array_equal(train_pred[0][ii][jj], y_train_batch[ii][jj]))
                        total = sum(train_end_idx[i * batch_size:(i + 1) * batch_size])[0]
                        acc2 = float(equal) / float(total)

                        print("Batch: %.f/%.f. train loss: %.4f, train accuracy: %.4f"
                              % ((i + 1) % n_train_batches, n_train_batches, batch_loss, acc2))
                        # else:
                        # print("Batch: %d/%d" % ((i + 1), n_train_batches))

                    if (i + 1) % 10 == 0 and not first_batch:
                        permutation = np.random.permutation(x_test.shape[0])
                        x_test = x_test[permutation]
                        y_test = y_test[permutation]
                        test_end_idx = test_end_idx[permutation]

                        for j in range(1):
                            equal = 0
                            total = 0
                            x_test_batch = x_test[j * batch_size:(j + 1) * batch_size]
                            y_test_batch = y_test[j * batch_size:(j + 1) * batch_size]

                            # acc1 = sess.run([acc], feed_dict={self.X: x_test_batch, self.Y: y_test_batch})
                            test_pred = sess.run([prediction], feed_dict={self.X: x_test_batch})
                            test_loss = sess.run([self.loss], feed_dict={self.X: x_test_batch, self.Y: y_test_batch})

                            for ii in range(len(test_pred[0])):
                                for jj in range(
                                        len(test_pred[0][ii][
                                            :test_end_idx[j * batch_size:(j + 1) * batch_size][ii][0]])):
                                    equal += int(np.array_equal(test_pred[0][ii][jj], y_test_batch[ii][jj]))
                            total += sum(test_end_idx[j * batch_size:(j + 1) * batch_size])[0]
                            acc2 = float(equal) / float(total)

                            print("Batch: %.f/%.f. test loss: %.4f, test accuracy: %.4f" %
                                  ((j + 1) % n_test_batches, 1, test_loss[0], acc2))

                    first_batch = False

                print("Epoch: %d. Epoch loss: %.4f " % (epoch, epoch_loss))

                # Prediction step
                if (epoch + 1) % 1 == 0  and not first_epoch:
                    print("Prediction on validation set...")
                    saver.save(sess, "output/checkpoints/", epoch)

                    equal = 0
                    total = 0
                    prediction_array = []
                    for i in range(n_valid_batches):
                        x_valid_batch = x_valid[i * batch_size:(i + 1) * batch_size]
                        y_valid_batch = y_valid[i * batch_size:(i + 1) * batch_size]

                        valid_pred = sess.run([prediction], feed_dict={self.X: x_valid_batch})
                        prediction_array.append(valid_pred[0])

                        for ii in range(len(valid_pred[0])):
                            for jj in range(
                                    len(valid_pred[0][ii][
                                        :valid_end_idx[i * batch_size:(i + 1) * batch_size][ii][0]])):
                                equal += int(np.array_equal(valid_pred[0][ii][jj], y_valid_batch[ii][jj]))
                        total += sum(valid_end_idx[i * batch_size:(i + 1) * batch_size])[0]

                    prediction_acc = float(equal) / float(total)
                    print("Prediction accuracy: %.4f" % prediction_acc)
                    np.save("output/prediction/pred-at-epoch-" + str(epoch), np.array(prediction_array))

                first_epoch = False

