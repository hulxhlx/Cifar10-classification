
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
from keras.layers import BatchNormalization
from keras.layers import Conv2D, Flatten, Dense,merge,Dropout, Add, Lambda, \
    Activation, AveragePooling2D, MaxPooling2D, Concatenate
from keras import regularizers, initializers
from keras import backend as K

tf.disable_v2_behavior()

class GOOGLESVM:
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
        self.name = "GOOGLESVM"
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

            num_A_blocks = 1
            num_B_blocks = 1
            num_C_blocks = 1
            x = self.inception_v4_stem(x_input)
            for i in range(num_A_blocks):
              x = self.inception_v4_A(x)
              x = self.inception_v4_reduction_A(x)
            for i in range(num_B_blocks):
              x = self.inception_v4_B(x)
              x = self.inception_v4_reduction_B(x)
            for i in range(num_C_blocks):
              x = self.inception_v4_C(x)

            x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')(x)
            x = Dropout(0.5)(x)
            full_one_dropout = Flatten()(x)
            # full_one_dropout = GlobalAveragePooling2D(name='pool')(x)
            readout_weight = self.weight_variable([1536, num_classes])
            readout_bias = self.bias_variable([num_classes])
            output = tf.matmul(full_one_dropout, readout_weight) + readout_bias

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
        test_log_path = 'logs/test'
        train_log_path = 'logs/train'
        train_writer = tf.summary.FileWriter(
            logdir=train_log_path, graph=tf.get_default_graph()
        )
        test_writer = tf.summary.FileWriter(
            logdir=test_log_path
        )

        with tf.Session() as sess:
            sess.run(init)

            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)


            for index in range(epochs):
                # train by batch


                # every 100th step and at 0,
                if index % 400 == 0 and index != 0:
                    feed_dict = {
                        self.x_input: batch_features,
                        self.y_input: batch_labels,
                    }

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
                    }

                    summary, test_accuracy = sess.run([self.merged, self.test_accuracy], feed_dict=feed_dict)
                    test_writer.add_summary(summary=summary, global_step=index / 400)

                    print(
                        "step: {}, test accuracy : {}".format(
                            index/400, test_accuracy
                        )
                    )

                batch_features, batch_labels = train_data.next_batch(self.batch_size)
                batch_labels[batch_labels == 0] = -1

                    # input dictionary with dropout of 50%
                feed_dict = {
                        self.x_input: batch_features,
                        self.y_input: batch_labels,
                    }

                    # run the train op
                summary, _, loss = sess.run(
                        [self.merged, self.optimizer, self.loss], feed_dict=feed_dict
                    )
                train_writer.add_summary(summary=summary, global_step=index/400)


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



    @staticmethod
    def normal_full_layer(input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        init_random_dist = tf.truncated_normal([input_size, size], stddev=0.1)
        W = tf.Variable(init_random_dist)
        init_bias_vals = tf.constant(0.1, shape=[size])
        b = tf.Variable(init_bias_vals)

        return tf.matmul(input_layer, W) + b

    @staticmethod
    def Cov2dbn(x, nb_filter, kernel_size, activation, strides, padding, data_format="channels_last",kernel_initializer="he_normal",
              ):
        x = Conv2D(nb_filter, kernel_size, strides = strides, data_format = data_format,kernel_initializer = kernel_initializer, padding = padding )(x)
        x = Activation(activation)(x)
        return x

    @staticmethod
    def inception_v4_stem(x):
        def Cov2dbn(x, nb_filter, kernel_size, activation, strides, padding, data_format="channels_last",
                    kernel_initializer="he_normal",
                    ):
            x = Conv2D(nb_filter, kernel_size, strides=strides, data_format=data_format,
                       kernel_initializer=kernel_initializer, padding=padding)(x)
            x = Activation(activation)(x)
            return x
        # in original inception-v4, conv stride is 2
        x = Cov2dbn(x, 32, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='valid')
        x = Cov2dbn(x, 32, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='valid')
        x = Cov2dbn(x, 64, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')

        # in original inception-v4, stride is 2
        a = MaxPooling2D((3, 3), strides=(1, 1), padding="valid", data_format="channels_last")(x)

        # in original inception-v4, conv stride is 2
        b = Cov2dbn(x, 96, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='valid')
        x = merge.concatenate([a, b], axis=-1)

        a = Cov2dbn(x, 64, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        a = Cov2dbn(a, 96, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='valid')
        b = Cov2dbn(x, 64, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        print("ashape")
        print(a.shape)
        b = Cov2dbn(b, 64, (7, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        b = Cov2dbn(b, 64, (1, 7), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        b = Cov2dbn(b, 96, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='valid')
        print("bshape")
        print(b.shape)
        x = merge.concatenate([a, b], axis=-1)

        # in original inception-v4, conv stride should be 2
        a = Cov2dbn(x, 192, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='valid')
        # in original inception-v4, stride is 2
        b = MaxPooling2D((3, 3), strides=(1, 1), data_format="channels_last", padding='valid')(x)
        x = merge.concatenate([a, b], axis=-1)

        return x

    @staticmethod
    def inception_v4_A(x):
        def Cov2dbn(x, nb_filter, kernel_size, activation, strides, padding, data_format="channels_last",
                    kernel_initializer="he_normal",
                    ):
            x = Conv2D(nb_filter, kernel_size, strides=strides, data_format=data_format,
                       kernel_initializer=kernel_initializer, padding=padding)(x)
            x = Activation(activation)(x)
            return x
        a = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last")(x)

        a = Cov2dbn(a, 96, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        b = Cov2dbn(x, 96, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')

        c = Cov2dbn(x, 64, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        c = Cov2dbn(c, 96, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        d = Cov2dbn(x, 64, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        d = Cov2dbn(d, 96, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        d = Cov2dbn(d, 96, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        x = merge.concatenate([a, b, c, d], axis=-1)

        return x



    @staticmethod
    def inception_v4_reduction_A(x):
        def Cov2dbn(x, nb_filter, kernel_size, activation, strides, padding, data_format="channels_last",
                    kernel_initializer="he_normal",
                    ):
            x = Conv2D(nb_filter, kernel_size, strides=strides, data_format=data_format,
                       kernel_initializer=kernel_initializer, padding=padding)(x)
            x = Activation(activation)(x)
            return x
        a = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
        b = Cov2dbn(x, 384, (3, 3), activation="relu", strides=(2, 2), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='valid')
        c = Cov2dbn(x, 192, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        c = Cov2dbn(c, 224, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        c = Cov2dbn(c, 256, (3, 3), activation="relu", strides=(2, 2), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='valid')

        x = merge.concatenate([a, b, c], axis=-1)

        return x


    @staticmethod
    def inception_v4_B(x):
        def Cov2dbn(x, nb_filter, kernel_size, activation, strides, padding, data_format="channels_last",
                    kernel_initializer="he_normal",
                    ):
            x = Conv2D(nb_filter, kernel_size, strides=strides, data_format=data_format,
                       kernel_initializer=kernel_initializer, padding=padding)(x)
            x = Activation(activation)(x)
            return x
        a = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        a = Cov2dbn(a, 128, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')

        b = Cov2dbn(x, 384, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        c = Cov2dbn(x, 192, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        c = Cov2dbn(c, 224, (1, 7), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        c = Cov2dbn(c, 256, (1, 7), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')

        d = Cov2dbn(x, 192, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        d = Cov2dbn(d, 192, (1, 7), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        d = Cov2dbn(d, 224, (7, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        d = Cov2dbn(d, 224, (1, 7), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        d = Cov2dbn(d, 256, (7, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')

        x = merge.concatenate([a, b, c, d], axis=-1)

        return x


    @staticmethod
    def inception_v4_reduction_B(x):
        def Cov2dbn(x, nb_filter, kernel_size, activation, strides, padding, data_format="channels_last",
                    kernel_initializer="he_normal",
                    ):
            x = Conv2D(nb_filter, kernel_size, strides=strides, data_format=data_format,
                       kernel_initializer=kernel_initializer, padding=padding)(x)
            x = Activation(activation)(x)
            return x
        a = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
        b = Cov2dbn(x, 192, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        b = Cov2dbn(b, 192, (3, 3), activation="relu", strides=(2, 2), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='valid')
        c = Cov2dbn(x, 256, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        c = Cov2dbn(c, 256, (1, 7), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        c = Cov2dbn(c, 256, (7, 1), activation="relu", strides=(1, 1), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='same')
        c = Cov2dbn(c, 320, (3, 3), activation="relu", strides=(2, 2), data_format="channels_last",
                    kernel_initializer="he_normal",
                    padding='valid')

        x = merge.concatenate([a, b, c], axis=-1)

        return x

    @staticmethod
    def inception_v4_C(x):
        def Cov2dbn(x, nb_filter, kernel_size, activation, strides, padding, data_format="channels_last",
                    kernel_initializer="he_normal",
                    ):
            x = Conv2D(nb_filter, kernel_size, strides=strides, data_format=data_format,
                       kernel_initializer=kernel_initializer, padding=padding)(x)
            x = Activation(activation)(x)
            return x
        a = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        a = Cov2dbn(a, 256, (1, 1), strides=(1, 1), activation='relu',
                    kernel_initializer='he_normal', padding='same')

        b = Cov2dbn(x, 256, (1, 1), strides=(1, 1), activation='relu',
                    kernel_initializer='he_normal', padding='same')

        c = Cov2dbn(x, 384, (1, 1), strides=(1, 1), activation='relu',
                    kernel_initializer='he_normal', padding='same')
        c1 = Cov2dbn(c, 256, (1, 3), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same')
        c2 = Cov2dbn(c, 256, (3, 1), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same')

        d = Cov2dbn(x, 384, (1, 1), strides=(1, 1), activation='relu',
                    kernel_initializer='he_normal', padding='same')
        d = Cov2dbn(d, 448, (1, 3), strides=(1, 1), activation='relu',
                    kernel_initializer='he_normal', padding='same')
        d = Cov2dbn(d, 512, (3, 1), strides=(1, 1), activation='relu',
                    kernel_initializer='he_normal', padding='same')
        d1 = Cov2dbn(d, 256, (3, 1), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same')
        d2 = Cov2dbn(d, 256, (1, 3), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same')

        x = merge.concatenate([a, b, c1, c2, d1, d2], axis=-1)

        return x