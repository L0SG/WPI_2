import tensorflow as tf
import numpy as np

class vgg11:
    """
    the standard vgg11 model
    should contain images, labels, weights and session
    """
    def __init__(self, weights=None, sess=None):
        """
        initialize & build the model
        :param weights: if not none, load pre-computed weights
        :param sess: pass the session defined from the main script
        """
        if weights is not None:
            self.weights = weights
            print ("VGG-11 pre-computed weights loaded")
        if sess is not None:
            self.sess = sess
        # build the model
        self.model = self.build()


    # anna
    def build(self):
        """
        build the model
        the model should output softmax class prediction tensor
        :return: vgg11 model
        """
        # impelement here
        self.conv3_64 = self.conv_layer(self.images, 'con3_64', 3, 64)
        self.pool1 = self.max_pool(self.conv3_64, 'pool1')

        self.conv3_128 = self.conv_layer(self.pool1, 'conv3_128', 64, 128)
        self.pool2 = self.max_pool(self.conv3_128, 'pool2')

        self.conv3_256_1 = self.conv_layer(self.pool2, 'conv3_256_1', 128, 256)
        self.conv3_256_2 = self.conv_layer(self.conv3_256_1, 'conv3_256_2', 256, 256)
        self.pool3 = self.max_pool(self.conv3_256_2, 'pool3')

        self.conv3_512_1 = self.conv_layer(self.pool3, 'conv3_512_1', 256, 512)
        self.conv3_512_2 = self.conv_layer(self.conv3_512_1, 'conv3_512_2', 512, 512)
        self.pool4 = self.max_pool(self.conv3_512_2, 'pool4')

        self.conv3_512_3 = self.conv_layer(self.pool4, 'conv3_512_3', 512, 512)
        self.conv3_512_4 = self.conv_layer(self.conv3_512_3, 'conv3_512_4', 512, 512)
        self.pool5 = self.max_pool(self.conv3_512_4, 'pool5')

        shape = int(np.prod(self.pool5.get_shape()[1:]))
        self.fc4096_1 = self.fc_layer(self.pool1, 'fc4096_1', shape, 4096)
        self.fc4096_2 = self.fc_layer(self.fc4096_1, 'fc4096_2', 4096, 4096)
        self.fc1000 = self.fc_layer(self.fc4096_2, 'fc1000', 4096, 1000)

        model = tf.nn.softmax(self.fc1000, name='softmax_out')

        return model

    def fc_bias(self, name, out_size):
        return tf.Variable(tf.constant(1.0, shape=[out_size], dtype='float32'), name=name)

    def fc_weight(self, name, in_size, out_size):
        return tf.Variable(tf.truncated_normal([in_size, out_size], dtype='float32', stddev=1e-1), name=name)

    def fc_layer(self, input, name, in_size, out_size):
        with tf.variable_scope(name):
            weight = self.fc_weight(name, in_size, out_size)

            bias = self.fc_bias(name, out_size)

            flatten = tf.reshape(input, [-1, in_size])

            fc = tf.nn.bias_add(tf.matmul(flatten, weight), bias)

            return tf.nn.relue(fc)

    def max_pool(self, input, name):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_channel(self, name, pre_num_ch, num_ch):
        return tf.Variable(tf.truncated_normal([3, 3, pre_num_ch, num_ch], dtype='float32', stddev=1e-1), name=name)

    def conv_bias(self, name, num_ch):
        return tf.Variable(tf.constant(0.0, shape=[num_ch], dtype='float32'), name=name)

    def conv_layer(self, input, name, pre_num_ch, num_ch):
        with tf.variable_scope(name):
            channel = self.conv_channel(name, pre_num_ch, num_ch)

            conv = tf.nn.conv2d(input, channel, [1, 1, 1, 1],
                                padding='SAME')  # VAlID = without padding, SAME = with zero padding

            conv_biases = self.conv_bias(name, num_ch)

            out = tf.nn.bias_add(conv, conv_biases)
            return tf.nn.relu(out)

    def train(self, images, labels, epochs, val_split, save_weights):
        """
        train the model with given epochs and val_split
        :param images: image batch tensor for placeholder x
        :param labels: image labels tensor for placeholder y
        :param epochs: num of epochs
        :param val_split: validation split (from 0 to 1)
        :param save_weights: save the weights if true
        """
        # impelement here


    def predict(self, images):
        """
        predict the class given the images
        :param images: the test set image
        :return: list with shape [None, 1], where None is sample size and 1 is class num integer
        """
        # implement here
        preds = None
        return preds