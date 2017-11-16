"""
This file dedicates to configure models (Tensorflow)
1. create model
2. load parameters
3. prune filters and fc nodes
4. save pruned model
"""
import tensorflow as tf
import inspect
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time

batch_size = 40
num_classes = 10
height_width = 224
# checkpoint initialization
checkpoint_dir = './models_tf/'
model_tag = 'VGG_10p'


class VGG:
    def __init__(self, conv_setting, recover_setting, weights=None, freeze=True, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.x = tf.placeholder(tf.float32, shape=[None, height_width, height_width, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, num_classes])
        ''' Switch between framework and preload'''
        logits, self.kp = VGG_preload(self.x, conv_setting, weights, recover_setting, freeze=freeze)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(logits, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(5e-5).minimize(self.loss)
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, train_generator, validation_generator, epochs, train_steps, validation_steps):
        saver = tf.train.Saver(max_to_keep=1)
        load(saver, self.sess, checkpoint_dir)
        val_acc = 0
        for j in range(epochs):
            # train
            loss = 0
            acc = 0
            for i in range(train_steps):
                batch = train_generator.next()
                _, l, a = self.sess.run([self.train_op, self.loss, self.accuracy], feed_dict={
                    self.x: batch[0],
                    self.y: batch[1],
                    self.kp: self.keep_prob
                })
                loss += l
                acc += a
                print("\rEpoch = %d/%d; Progress = %d/%d; loss = %.3f; acc = %.4f" % (
                    j + 1, epochs, (i + 1), train_steps, l, a), end='')

            print("\rEpoch = %d/%d; loss = %.3f; acc = %.4f" % (
                j + 1, epochs, loss / train_steps, acc / train_steps))

            # validate
            val_acc_cur = self.validate(validation_generator, validation_steps)
            # store
            if val_acc_cur > val_acc:
                val_acc = val_acc_cur
                save(saver, self.sess, checkpoint_dir, j, model_tag)

    def validate(self, validation_generator, validation_steps):
        # validate
        val_loss = 0
        val_acc = 0
        for i in range(validation_steps):
            batch = validation_generator.next()
            l, a = self.sess.run([self.loss, self.accuracy], feed_dict={
                self.x: batch[0],
                self.y: batch[1],
                self.kp: 1.0
            })
            val_loss += l
            val_acc += a
        print("val_loss = %.3f; val_acc = %.4f" % (
            val_loss / validation_steps, val_acc / validation_steps))
        return val_acc


def VGG_framework(x, conv_setting):
    """
    General helper function for VGG construction
    Reserve for training from scratch -- baseline
    """
    # block1: conv x 2 + pool
    with tf.name_scope('block1'):
        W_conv1 = weight_variable([3, 3, 3, conv_setting[0]])  # w x h x channel x map
        b_conv1 = bias_variable([conv_setting[0]])
        layer = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

        W_conv2 = weight_variable([3, 3, conv_setting[0], conv_setting[1]])
        b_conv2 = bias_variable([conv_setting[1]])
        layer = tf.nn.relu(conv2d(layer, W_conv2) + b_conv2)
        layer = max_pool_2x2(layer)

    # block2: conv x 2 + pool
    with tf.name_scope('block2'):
        W_conv3 = weight_variable([3, 3, conv_setting[1], conv_setting[2]])  # w x h x channel x map
        b_conv3 = bias_variable([conv_setting[2]])
        layer = tf.nn.relu(conv2d(layer, W_conv3) + b_conv3)

        W_conv4 = weight_variable([3, 3, conv_setting[2], conv_setting[3]])
        b_conv4 = bias_variable([conv_setting[3]])
        layer = tf.nn.relu(conv2d(layer, W_conv4) + b_conv4)
        layer = max_pool_2x2(layer)

    # block3: conv x 3 + pool
    with tf.name_scope('block3'):
        W_conv5 = weight_variable([3, 3, conv_setting[3], conv_setting[4]])  # w x h x channel x map
        b_conv5 = bias_variable([conv_setting[4]])
        layer = tf.nn.relu(conv2d(layer, W_conv5) + b_conv5)

        W_conv6 = weight_variable([3, 3, conv_setting[4], conv_setting[5]])
        b_conv6 = bias_variable([conv_setting[5]])
        layer = tf.nn.relu(conv2d(layer, W_conv6) + b_conv6)

        W_conv7 = weight_variable([3, 3, conv_setting[5], conv_setting[6]])
        b_conv7 = bias_variable([conv_setting[6]])
        layer = tf.nn.relu(conv2d(layer, W_conv7) + b_conv7)
        layer = max_pool_2x2(layer)

    # block4: conv x 3 + pool
    with tf.name_scope('block4'):
        W_conv8 = weight_variable([3, 3, conv_setting[6], conv_setting[7]])  # w x h x channel x map
        b_conv8 = bias_variable([conv_setting[7]])
        layer = tf.nn.relu(conv2d(layer, W_conv8) + b_conv8)

        W_conv9 = weight_variable([3, 3, conv_setting[7], conv_setting[8]])
        b_conv9 = bias_variable([conv_setting[8]])
        layer = tf.nn.relu(conv2d(layer, W_conv9) + b_conv9)

        W_conv10 = weight_variable([3, 3, conv_setting[8], conv_setting[9]])
        b_conv10 = bias_variable([conv_setting[9]])
        layer = tf.nn.relu(conv2d(layer, W_conv10) + b_conv10)
        layer = max_pool_2x2(layer)

    # block5: conv x 3 + pool
    with tf.name_scope('block5'):
        W_conv11 = weight_variable([3, 3, conv_setting[9], conv_setting[10]])  # w x h x channel x map
        b_conv11 = bias_variable([conv_setting[10]])
        layer = tf.nn.relu(conv2d(layer, W_conv11) + b_conv11)

        W_conv12 = weight_variable([3, 3, conv_setting[10], conv_setting[11]])
        b_conv12 = bias_variable([conv_setting[11]])
        layer = tf.nn.relu(conv2d(layer, W_conv12) + b_conv12)

        W_conv13 = weight_variable([3, 3, conv_setting[11], conv_setting[12]])
        b_conv13 = bias_variable([conv_setting[12]])
        layer = tf.nn.relu(conv2d(layer, W_conv13) + b_conv13)
        layer = max_pool_2x2(layer)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * conv_setting[12], 512])
        b_fc1 = bias_variable([512])
        layer = tf.reshape(layer, [-1, 7 * 7 * conv_setting[12]])
        h_fc1 = tf.nn.relu(tf.matmul(layer, W_fc1) + b_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([512, num_classes])
        b_fc2 = bias_variable([num_classes])
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv


def VGG_preload(x, conv_setting, weights, recover_setting, freeze=True):
    """
    General helper function for VGG construction
    Pre-load parameter (freeze)
    Recover according to recover_setting
    :param x:
    :param conv_setting: original model parameter setting
    :param weights: weights: list (1-13 + 14-15; conv + fc) of tuple (W, b)
    :param recover_setting: new model parameter setting
    :param freeze: do we want to freeze the preload
    :return:
    """
    lrs = [recover_setting[i] - conv_setting[i] for i in range(len(recover_setting))]  # list of recover setting
    # block1: conv x 2 + pool
    with tf.name_scope('block1'):
        W_conv1 = tf.Variable(weights[0][0], trainable=freeze)
        b_conv1 = tf.Variable(weights[0][1], trainable=freeze)
        W_conv1, b_conv1 = recover_filter(W_conv1, b_conv1, 0, lrs[0])
        layer = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

        W_conv2 = tf.Variable(weights[1][0], trainable=freeze)
        b_conv2 = tf.Variable(weights[1][1], trainable=freeze)
        W_conv2, b_conv2 = recover_filter(W_conv2, b_conv2, lrs[0], lrs[1])
        layer = tf.nn.relu(conv2d(layer, W_conv2) + b_conv2)
        layer = max_pool_2x2(layer)

    # block2: conv x 2 + pool
    with tf.name_scope('block2'):
        W_conv3 = tf.Variable(weights[2][0], trainable=freeze)
        b_conv3 = tf.Variable(weights[2][1], trainable=freeze)
        W_conv3, b_conv3 = recover_filter(W_conv3, b_conv3, lrs[1], lrs[2])
        layer = tf.nn.relu(conv2d(layer, W_conv3) + b_conv3)

        W_conv4 = tf.Variable(weights[3][0], trainable=freeze)
        b_conv4 = tf.Variable(weights[3][1], trainable=freeze)
        W_conv4, b_conv4 = recover_filter(W_conv4, b_conv4, lrs[2], lrs[3])
        layer = tf.nn.relu(conv2d(layer, W_conv4) + b_conv4)
        layer = max_pool_2x2(layer)

    # block3: conv x 3 + pool
    with tf.name_scope('block3'):
        W_conv5 = tf.Variable(weights[4][0], trainable=freeze)
        b_conv5 = tf.Variable(weights[4][1], trainable=freeze)
        W_conv5, b_conv5 = recover_filter(W_conv5, b_conv5, lrs[3], lrs[4])
        layer = tf.nn.relu(conv2d(layer, W_conv5) + b_conv5)

        W_conv6 = tf.Variable(weights[5][0], trainable=freeze)
        b_conv6 = tf.Variable(weights[5][1], trainable=freeze)
        W_conv6, b_conv6 = recover_filter(W_conv6, b_conv6, lrs[4], lrs[5])
        layer = tf.nn.relu(conv2d(layer, W_conv6) + b_conv6)

        W_conv7 = tf.Variable(weights[6][0], trainable=freeze)
        b_conv7 = tf.Variable(weights[6][1], trainable=freeze)
        W_conv7, b_conv7 = recover_filter(W_conv7, b_conv7, lrs[5], lrs[6])
        layer = tf.nn.relu(conv2d(layer, W_conv7) + b_conv7)
        layer = max_pool_2x2(layer)

    # block4: conv x 3 + pool
    with tf.name_scope('block4'):
        W_conv8 = tf.Variable(weights[7][0], trainable=freeze)
        b_conv8 = tf.Variable(weights[7][1], trainable=freeze)
        W_conv8, b_conv8 = recover_filter(W_conv8, b_conv8, lrs[6], lrs[7])
        layer = tf.nn.relu(conv2d(layer, W_conv8) + b_conv8)

        W_conv9 = tf.Variable(weights[8][0], trainable=freeze)
        b_conv9 = tf.Variable(weights[8][1], trainable=freeze)
        W_conv9, b_conv9 = recover_filter(W_conv9, b_conv9, lrs[7], lrs[8])
        layer = tf.nn.relu(conv2d(layer, W_conv9) + b_conv9)

        W_conv10 = tf.Variable(weights[9][0], trainable=freeze)
        b_conv10 = tf.Variable(weights[9][1], trainable=freeze)
        W_conv10, b_conv10 = recover_filter(W_conv10, b_conv10, lrs[8], lrs[9])
        layer = tf.nn.relu(conv2d(layer, W_conv10) + b_conv10)
        layer = max_pool_2x2(layer)

    # block5: conv x 3 + pool
    with tf.name_scope('block5'):
        W_conv11 = tf.Variable(weights[10][0], trainable=freeze)
        b_conv11 = tf.Variable(weights[10][1], trainable=freeze)
        W_conv11, b_conv11 = recover_filter(W_conv11, b_conv11, lrs[9], lrs[10])
        layer = tf.nn.relu(conv2d(layer, W_conv11) + b_conv11)

        W_conv12 = tf.Variable(weights[11][0], trainable=freeze)
        b_conv12 = tf.Variable(weights[11][1], trainable=freeze)
        W_conv12, b_conv12 = recover_filter(W_conv12, b_conv12, lrs[10], lrs[11])
        layer = tf.nn.relu(conv2d(layer, W_conv12) + b_conv12)

        W_conv13 = tf.Variable(weights[12][0], trainable=freeze)
        b_conv13 = tf.Variable(weights[12][1], trainable=freeze)
        W_conv13, b_conv13 = recover_filter(W_conv13, b_conv13, lrs[11], lrs[12])
        layer = tf.nn.relu(conv2d(layer, W_conv13) + b_conv13)
        layer = max_pool_2x2(layer)

    with tf.name_scope('fc1'):
        W_fc1 = tf.Variable(weights[13][0], trainable=freeze)
        b_fc1 = tf.Variable(weights[13][1], trainable=freeze)
        W_fc1, b_fc1 = recover_fc(W_fc1, b_fc1, lrs[12])
        layer = tf.reshape(layer, [-1, 7 * 7 * recover_setting[12]])
        h_fc1 = tf.nn.relu(tf.matmul(layer, W_fc1) + b_fc1)
        # dropout here
        keep_prob = tf.placeholder(tf.float32)
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = tf.Variable(weights[14][0])
        b_fc2 = tf.Variable(weights[14][1])
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv, keep_prob


def recover_filter(W, b, map_plus, filter_plus):
    """
    Take in W and b (usually untrainable)
    Return new filter with map_plus and filter_plus indicating the recovery setting
    :param W: old weight
    :param b: old bias
    :param map_plus: in 3rd dimension. The number of increased feature maps of last layer
    :param filter_plus: in 4th dimension. The number of increased feature maps of current layer
    :return: new W and b
    """
    W = tf.concat([W, weight_variable([3, 3, map_plus, int(W.shape[3])])], 2) if map_plus > 0 else W
    W = tf.concat([W, weight_variable([3, 3, int(W.shape[2]), filter_plus])], 3) if filter_plus > 0 else W
    b = tf.concat([b, bias_variable([filter_plus])], 0) if filter_plus > 0 else b
    return W, b


def recover_fc(W, b, map_plus):
    W = tf.reshape(W, [7, 7, int(W.shape[0]) // 49, 512])
    W = tf.concat([W, weight_variable([7, 7, map_plus, 512])], 2)
    W = tf.reshape(W, [-1, 512])
    return W, b


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 down samples a feature map by 2X."""
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


def save(saver, sess, ckpt_dir, step, model_name):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    saver.save(sess,
               os.path.join(ckpt_dir, model_name),
               global_step=step)


def load(saver, sess, ckpt_dir):
    print("Reading ckpt...")

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    else:
        return False
