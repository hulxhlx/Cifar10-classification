# Copyright 2017-2020 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""2 Convolutional Layers with Max Pooling CNN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import os
import tensorflow.compat.v1 as tf
import time
import sys
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Add, Lambda, \
    Activation, GlobalAveragePooling2D, MaxPooling2D, Concatenate
from keras import regularizers, initializers
from keras import backend as K
tf.disable_v2_behavior()

class RESNETSVM:
    def __init__(self, alpha, batch_size, num_classes, num_features, penalty_parameter):
        """Initializes the CNN-SVM model

        :param alpha: The learning rate to be used by the model.
        :param batch_size: The number of batches to use for training/validation/testing.
        :param num_classes: The number of classes in the dataset.
        :param num_features: The number of features in the dataset.
        :param penalty_parameter: The SVM C penalty parameter.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.name = "CNN-SVM"
        self.num_classes = num_classes
        self.num_features = num_features
        self.penalty_parameter = penalty_parameter

        def __graph__():

            with tf.name_scope("input"):
                # [BATCH_SIZE, NUM_FEATURES]
                x_input = tf.placeholder(
                    dtype=tf.float32, shape=[None, 32,32,3], name="x_input"
                )

                # [BATCH_SIZE, NUM_CLASSES]
                y_input = tf.placeholder(
                    dtype=tf.float32, shape=[None, num_classes], name="actual_label"
                )
            convo_1 = self.convolutional_layer(x_input, shape=[4, 4, 3, 32])
            convo_1_pooling = self.max_pool_2by2(convo_1)
            convo_2 = self.convolutional_layer(convo_1_pooling, shape=[4, 4, 32, 64])
            convo_2_pooling = self.max_pool_2by2(convo_2)
            convo_2_flat = tf.reshape(convo_2_pooling, [-1, 8 * 8 * 64])
            keep_prob = tf.placeholder(tf.float32)
            full_layer_one = tf.nn.relu(self.normal_full_layer(convo_2_flat, 1024))
            full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=keep_prob)
            readout_weight = self.weight_variable([1024, num_classes])
            readout_bias = self.bias_variable([num_classes])
            output = tf.matmul(full_one_dropout, readout_weight) + readout_bias
            # x = self.conv_bn_relu(x_input, 16, (3, 3), (1, 1), 'lv0')
            # num_blocks = 6
            # # level 1:
            # # input: 32x32x16; output: 32x32x16
            # for i in range(num_blocks):
            #     x = self.res_block(x, 16, name='lv1_blk{}'.format(i + 1))
            #
            # # level 2:
            # # input: 32x32x16; output: 16x16x32
            # for i in range(num_blocks):
            #     x = self.res_block(x, 32, name='lv2_blk{}'.format(i + 1))
            #
            # # level 3:
            # # input: 16x16x32; output: 8x8x64
            # for i in range(num_blocks):
            #     x = self.res_block(x, 64, name='lv3_blk{}'.format(i + 1))
            #
            # # output
            # full_one_dropout = GlobalAveragePooling2D(name='pool')(x)
            # readout_weight = self.weight_variable([64, num_classes])
            # readout_bias = self.bias_variable([num_classes])
            # output = tf.matmul(full_one_dropout, readout_weight) + readout_bias

            with tf.name_scope("svm"):
                regularization_loss = tf.reduce_mean(tf.square(readout_weight))
                hinge_loss = tf.reduce_mean(
                    tf.square(
                        tf.maximum(
                            tf.zeros([batch_size, num_classes]), 1 - y_input * output
                        )
                    )
                )
                with tf.name_scope("loss"):
                    loss = regularization_loss + penalty_parameter * hinge_loss

            tf.summary.scalar("loss", loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)


            with tf.name_scope("accuracy"):
                output = tf.identity(tf.sign(output), name="prediction")
                correct_prediction = tf.equal(
                    tf.argmax(output, 1), tf.argmax(y_input, 1)
                )
                with tf.name_scope("accuracy"):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

            with tf.name_scope("train_accuracy"):
                output = tf.identity(tf.sign(output), name="prediction")
                correct_prediction = tf.equal(
                    tf.argmax(output, 1), tf.argmax(y_input, 1)
                )
                with tf.name_scope("train_accuracy"):
                    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("train_accuracy", train_accuracy)

            with tf.name_scope("test_accuracy"):
                output = tf.identity(tf.sign(output), name="prediction")
                correct_prediction = tf.equal(
                    tf.argmax(output, 1), tf.argmax(y_input, 1)
                )
                with tf.name_scope("test_accuracy"):
                    test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("test_accuracy", test_accuracy)

            merged = tf.summary.merge_all()
            # self.testshape = testshape
            self.x_input = x_input
            self.y_input = y_input
            self.keep_prob = 0.5
            self.output = output
            self.loss = loss
            # self.test_loss = test_loss
            self.optimizer = optimizer
            self.accuracy = accuracy
            self.merged = merged
            self.test_accuracy = test_accuracy
            self.train_accuracy = train_accuracy

        sys.stdout.write("\n<log> Building graph...")
        __graph__()
        sys.stdout.write("</log>\n")

    def train(self, checkpoint_path, epochs, log_path, train_data, test_data):
        """Trains the initialized model.

        :param checkpoint_path: The path where to save the trained model.
        :param epochs: The number of passes through the entire dataset.
        :param log_path: The path where to save the TensorBoard logs.
        :param train_data: The training dataset.
        :param test_data: The testing dataset.
        :return: None
        """

        if not os.path.exists(path=log_path):
            os.mkdir(log_path)

        if not os.path.exists(path=checkpoint_path):
            os.mkdir(checkpoint_path)

        saver = tf.train.Saver(max_to_keep=4)

        init = tf.global_variables_initializer()

        timestamp = str(time.asctime())

        train_writer = tf.summary.FileWriter(
            logdir=log_path, graph=tf.get_default_graph()
        )

        with tf.Session() as sess:
            sess.run(init)

            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

            # if checkpoint and checkpoint.model_checkpoint_path:
            #     saver = tf.train.import_meta_graph(
            #         checkpoint.model_checkpoint_path + ".meta"
            #     )
            #     saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

            for index in range(epochs):
                # train by batch
                batch_features, batch_labels = train_data.next_batch(self.batch_size)
                batch_labels[batch_labels == 0] = -1

                # input dictionary with dropout of 50%
                feed_dict = {
                    self.x_input: batch_features,
                    self.y_input: batch_labels,
                    self.keep_prob: 0.5,
                }

                # run the train op
                summary, _, loss = sess.run(
                    [self.merged, self.optimizer, self.loss], feed_dict=feed_dict
                )

                # every 100th step and at 0,
                if index % 400 == 0 and index != 0:
                    feed_dict = {
                        self.x_input: batch_features,
                        self.y_input: batch_labels,
                        self.keep_prob: 1.0,
                    }

                    # get the accuracy of training
                    # train_accuracy,loss = sess.run([self.accuracy,self.loss], feed_dict=feed_dict)
                    train_accuracy = sess.run(self.train_accuracy, feed_dict=feed_dict)
                    # display the training accuracy
                    print(
                        "step: {}, training accuracy : {}, training loss : {}".format(
                            index/400, train_accuracy, loss
                        )
                    )


                    saver.save(
                        sess,
                        save_path=os.path.join(checkpoint_path, self.name),
                        global_step=index,
                    )
                    test_features = test_data.images
                    test_labels = test_data.labels
                    test_labels[test_labels == 0] = -1

                    feed_dict = {
                self.x_input: test_data.images,
                self.y_input: test_data.labels,
                self.keep_prob: 1.0,
                    }

                    test_accuracy = sess.run(self.test_accuracy, feed_dict=feed_dict)

                    print(
                        "step: {}, test accuracy : {}".format(
                            index/400, test_accuracy
                        )
                    )
                    # train_writer.add_summary(summary=test_accuracy, global_step=index/400)
                    train_writer.add_summary(summary=summary, global_step=index/400)

            # test_features = test_data.images
            # test_labels = test_data.labels
            # test_labels[test_labels == 0] = -1
            #
            # feed_dict = {
            #     self.x_input: test_features,
            #     self.y_input: test_labels,
            #     self.keep_prob: 1.0,
            # }
            #
            # test_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)

            # print("Test Accuracy: {}".format(test_accuracy))

    @staticmethod
    def weight_variable(shape):
        """Returns a weight matrix consisting of arbitrary values.

        :param shape: The shape of the weight matrix to create.
        :return: The weight matrix consisting of arbitrary values.
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """Returns a bias matrix consisting of 0.1 values.

        :param shape: The shape of the bias matrix to create.
        :return: The bias matrix consisting of 0.1 values.
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    momentum = 0.9
    epsilon = 1e-5
    weight_decay = 1e-4

    @staticmethod
    def conv_layer(x, filters, kernel_size, strides, name):
        return Conv2D(
            filters, kernel_size,
    strides=strides, padding='same',
    use_bias=False,
  kernel_initializer=initializers.he_normal(),
    kernel_regularizer=regularizers.l2(1e-4),
    name=name    )(x)

    @staticmethod
    def conv_bn_relu(x, filters, kernel_size, strides, name):
        def conv_layer(x, filters, kernel_size, strides, name):
            return Conv2D(
                filters, kernel_size,
                strides=strides, padding='same',
                use_bias=False,
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(1e-4),
                name=name)(x)
        x = conv_layer(x, filters, kernel_size, strides, name + '_conv2D')
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=name + '_BN')(x)
        x = Activation(activation='relu', name=name + '_relu')(x)
        return x
    @staticmethod
    def conv_bn(x, filters, kernel_size, strides, name):
        """conv2D block without activation"""

        def conv_layer(x, filters, kernel_size, strides, name):
            return Conv2D(
                filters, kernel_size,
                strides=strides, padding='same',
                use_bias=False,
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(1e-4),
                name=name)(x)
        x = conv_layer(x, filters, kernel_size, strides, name + '_conv2D')
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=name + '_BN')(x)
        return x
    @staticmethod
    def res_block(x, dim, name):
        """residue block: two 3X3 conv2D stacks"""

        def conv_bn_relu(x, filters, kernel_size, strides, name):
            def conv_layer(x, filters, kernel_size, strides, name):
                return Conv2D(
                    filters, kernel_size,
                    strides=strides, padding='same',
                    use_bias=False,
                    kernel_initializer=initializers.he_normal(),
                    kernel_regularizer=regularizers.l2(1e-4),
                    name=name)(x)

            x = conv_layer(x, filters, kernel_size, strides, name + '_conv2D')
            x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=name + '_BN')(x)
            x = Activation(activation='relu', name=name + '_relu')(x)
            return x

        def conv_bn(x, filters, kernel_size, strides, name):
            """conv2D block without activation"""
            momentum = 0.9
            epsilon = 1e-5
            weight_decay = 1e-4
            def conv_layer(x, filters, kernel_size, strides, name):
                return Conv2D(
                    filters, kernel_size,
                    strides=strides, padding='same',
                    use_bias=False,
                    kernel_initializer=initializers.he_normal(),
                    kernel_regularizer=regularizers.l2(1e-4),
                    name=name)(x)

            x = conv_layer(x, filters, kernel_size, strides, name + '_conv2D')
            x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_BN')(x)
            return x
        input_dim = int(x.shape[-1])

        # shortcut
        identity = x
        if input_dim != dim:  # option A in the original paper
            identity = MaxPooling2D(
                pool_size=(1, 1), strides=(2, 2),
                padding='same',
                name=name + '_shortcut_pool'
            )(identity)

            identity = Lambda(
                lambda y: K.concatenate([y, K.zeros_like(y)]),
                name=name + '_shortcut_zeropad'
            )(identity)

        # residual path
        res = x
        if input_dim != dim:
            res = conv_bn_relu(res, dim, (3, 3), (2, 2), name + '_res_conv1')
        else:
            res = conv_bn_relu(res, dim, (3, 3), (1, 1), name + '_res_conv1')

        res = conv_bn(res, dim, (3, 3), (1, 1), name + '_res_conv2')

        # add identity and residue path
        out = Add(name=name + '_add')([identity, res])
        out = Activation(activation='relu', name=name + '_relu')(out)
        return out




    @staticmethod
    def max_pool_2x2(features):
        """Downnsamples the image based on convolutional layer

        :param features: The input to downsample.
        :return: Downsampled input.
        """
        return tf.nn.max_pool(
            features, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )

    @staticmethod
    def convolutional_layer(input_x, shape):
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        W = tf.Variable(init_random_dist)
        init_bias_vals = tf.constant(0.1, shape=[shape[3]])
        b = tf.Variable(init_bias_vals)

        return tf.nn.relu(tf.nn.conv2d(input_x, W, strides=[1, 1, 1, 1], padding='SAME') + b)

    @staticmethod
    def max_pool_2by2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def normal_full_layer(input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        init_random_dist = tf.truncated_normal([input_size, size], stddev=0.1)
        W = tf.Variable(init_random_dist)
        init_bias_vals = tf.constant(0.1, shape=[size])
        b = tf.Variable(init_bias_vals)

        return tf.matmul(input_layer, W) + b
