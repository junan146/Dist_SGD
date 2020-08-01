# Most of the tensorflow code is adapted from Tensorflow's tutorial on using
# CNNs to train MNIST
# https://www.tensorflow.org/get_started/mnist/pros#build-a-multilayer-convolutional-network.  # noqa: E501

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging

import tf_variables


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class SimpleCNN(object):
    def __init__(self, dataset, batch_size=-1, learning_rate=1e-2, num_classes=10):
        self.dataset = dataset
        # probability of drop
        self.keep_prob = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [None, dataset.width, dataset.height, dataset.channels])
        self.y_ = tf.placeholder(tf.float32, [None, num_classes])
        # Build the graph for the deep net
        self.y_conv, self.endpoint = deepnn(
            self.x, num_classes=num_classes, dropout_keep_prob=self.keep_prob)
        logging.debug("shape of output: {}".format(self.y_conv))

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_, logits=self.y_conv)
        self.cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_step = self.optimizer.minimize(self.cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1),
                                          tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        self.sess = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                device_count={'GPU': 0})
        )
        self.sess.run(tf.global_variables_initializer())

        # Helper values.
        self.variables = tf_variables.TensorFlowVariables(
            self.cross_entropy, self.sess)
        self.grads = self.optimizer.compute_gradients(
            self.cross_entropy)
        self.grads_placeholder = [
            (tf.placeholder("float", shape=grad[1].get_shape()), grad[1])
            for grad in self.grads]
        self.apply_grads_placeholder = self.optimizer.apply_gradients(
            self.grads_placeholder)


    def compute_gradients(self, x, y):
        return self.sess.run([grad[0] for grad in self.grads],
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 0.5})

    def apply_gradients(self, gradients):
        feed_dict = {}
        for i in range(len(self.grads_placeholder)):
            feed_dict[self.grads_placeholder[i][0]] = gradients[i]
        self.sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

    def compute_loss_accuracy(self, x, y):
        return self.sess.run([self.cross_entropy, self.accuracy],
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 1.0})
