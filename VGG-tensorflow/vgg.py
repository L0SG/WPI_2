import tensorflow as tf
import time
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
        else :
            self.weights = {
                'wc1': tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1)),
                'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
                'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)),
                'wc4': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1)),
                'wc5': tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1)),
                'wc6': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'wc7': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'wc8': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'wd1': tf.Variable(tf.truncated_normal([512, 4096], stddev=0.1)),
                'wd2': tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1)),
                'wd3': tf.Variable(tf.truncated_normal([4096, 1000], stddev=0.1)),
                'wd4': tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1)),
                'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
                'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
                'bc3': tf.Variable(tf.random_normal([256], stddev=0.1)),
                'bc4': tf.Variable(tf.random_normal([256], stddev=0.1)),
                'bc5': tf.Variable(tf.random_normal([512], stddev=0.1)),
                'bc6': tf.Variable(tf.random_normal([512], stddev=0.1)),
                'bc7': tf.Variable(tf.random_normal([512], stddev=0.1)),
                'bc8': tf.Variable(tf.random_normal([512], stddev=0.1)),
                'bd1': tf.Variable(tf.random_normal([4096], stddev=0.1)),
                'bd2': tf.Variable(tf.random_normal([4096], stddev=0.1)),
                'bd3': tf.Variable(tf.random_normal([1000], stddev=0.1)),
                'bd4': tf.Variable(tf.random_normal([10], stddev=0.1))
            }
        if sess is not None:
            self.sess = sess
        # build the model
        self.model = self.build()


    def build(self):
        """
        build the model
        the model should output softmax class prediction tensor
        :return: vgg11 model
        """
        # impelement here
        self.x = tf.placeholder("float", [None, 32, 32, 3])
        self.y = tf.placeholder("float", [None, 100])

        model=self.conv(self.x,self.weights)
        print ("CNN READY")
        return model

    def conv(self, _input, _w):
        # CONV LAYER 1
        _conv1 = tf.nn.conv2d(_input, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        _mean, _var = tf.nn.moments(_conv1, [0, 1, 2])
        _conv1 = tf.nn.batch_normalization(_conv1, _mean, _var, 0, 1, 0.0001)
        _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _w['bc1']))
        _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # CONV LAYER 2
        _conv2 = tf.nn.conv2d(_pool1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        _mean, _var = tf.nn.moments(_conv2, [0, 1, 2])
        _conv2 = tf.nn.batch_normalization(_conv2, _mean, _var, 0, 1, 0.0001)
        _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _w['bc2']))
        _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # CONV LAYER 3&4
        _conv3 = tf.nn.conv2d(_pool2, _w['wc3'], strides=[1, 1, 1, 1], padding='SAME')
        _mean, _var = tf.nn.moments(_conv3, [0, 1, 2])
        _conv3 = tf.nn.batch_normalization(_conv3, _mean, _var, 0, 1, 0.0001)
        _conv3 = tf.nn.relu(tf.nn.bias_add(_conv3, _w['bc3']))
        _conv4 = tf.nn.conv2d(_conv3, _w['wc4'], strides=[1, 1, 1, 1], padding='SAME')
        _mean, _var = tf.nn.moments(_conv4, [0, 1, 2])
        _conv4 = tf.nn.batch_normalization(_conv4, _mean, _var, 0, 1, 0.0001)
        _conv4 = tf.nn.relu(tf.nn.bias_add(_conv4, _w['bc4']))
        _pool4 = tf.nn.max_pool(_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # CONV LAYER 5&6
        _conv5 = tf.nn.conv2d(_pool4, _w['wc5'], strides=[1, 1, 1, 1], padding='VALID')
        _mean, _var = tf.nn.moments(_conv5, [0, 1, 2])
        _conv5 = tf.nn.batch_normalization(_conv5, _mean, _var, 0, 1, 0.0001)
        _conv5 = tf.nn.relu(tf.nn.bias_add(_conv5, _w['bc5']))
        _conv6 = tf.nn.conv2d(_conv5, _w['wc6'], strides=[1, 1, 1, 1], padding='SAME')
        _mean, _var = tf.nn.moments(_conv6, [0, 1, 2])
        _conv6 = tf.nn.batch_normalization(_conv6, _mean, _var, 0, 1, 0.0001)
        _conv6 = tf.nn.relu(tf.nn.bias_add(_conv6, _w['bc6']))
        _pool6 = tf.nn.max_pool(_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # CONV LAYER 7&8
        _conv7 = tf.nn.conv2d(_pool6, _w['wc7'], strides=[1, 1, 1, 1], padding='SAME')
        _mean, _var = tf.nn.moments(_conv7, [0, 1, 2])
        _conv7 = tf.nn.batch_normalization(_conv7, _mean, _var, 0, 1, 0.0001)
        _conv7 = tf.nn.relu(tf.nn.bias_add(_conv7, _w['bc7']))
        _conv8 = tf.nn.conv2d(_conv7, _w['wc8'], strides=[1, 1, 1, 1], padding='SAME')
        _mean, _var = tf.nn.moments(_conv8, [0, 1, 2])
        _conv8 = tf.nn.batch_normalization(_conv8, _mean, _var, 0, 1, 0.0001)
        _conv8 = tf.nn.relu(tf.nn.bias_add(_conv8, _w['bc8']))
        _pool8 = tf.nn.max_pool(_conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # VECTORIZE
        _dense1 = tf.reshape(_pool8, [-1, _w['wd1'].get_shape().as_list()[0]])  # why not [-1,none]
        # FULLY CONNECTED LAYER 1
        _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _w['bd1']))
        _fc_dr1 = tf.nn.dropout(_fc1, 0.5)
        # FULLY CONNECTED LAYER 2
        _fc2 = tf.nn.relu(tf.add(tf.matmul(_fc_dr1, _w['wd2']), _w['bd2']))
        _fc_dr2 = tf.nn.dropout(_fc2, 0.5)
        # FULLY CONNECTED LAYER 3
        _fc3 = tf.add(tf.matmul(_fc_dr2, _w['wd3']), _w['bd3'])
        # SOFT-MAX
        _out = tf.nn.softmax(tf.add(tf.matmul(_fc3, _w['wd4']), _w['bd4']))
        # RETURN
        '''
        out = {'input': _input, 'conv1': _conv1, 'pool1': _pool1,
               'conv2': _conv2, 'pool2': _pool2,
               'conv3': _conv3, 'conv4': _conv4, 'pool4': _pool4,
               'conv5': _conv5, 'conv6': _conv6, 'pool6': _pool6,
               'conv7': _conv7, 'conv8': _conv8, 'pool8': _pool8,
               'dense1': _dense1, 'fc1': _fc1, 'fc_dr1': _fc_dr1, 'fc2': _fc2, 'fc_dr2': _fc_dr2,
               'fc3': _fc3,      'out': _out
               }
        '''
        return _out

    def train(self, images, labels, epochs, val_split, save_weights):
        """
        train the model with given epochs and val_split
        :param images: image batch tensor for placeholder x
        :param labels: image labels tensor for placeholder y
        :param epochs: num of epochs
        :param val_split: validation split (from 0 to 1)
        :param save_weights: save the weights if true
        """
        #1 data split
        data_size=images.shape[0]
        train_data_size = data_size*(1-val_split)
        test_data_size = data_size-data_size*(1-val_split)
        train_images, test_images=tf.split(images, [train_data_size, test_data_size])
        train_labels, test_labels=tf.split(labels, [train_data_size, test_data_size])

        x=self.x
        y=self.y
        sess=self.sess

        batch_size = 25
        total_batch = train_data_size/batch_size
        display_step = 10
        save_step = 100

        # impelement here
        pred = self.model
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        optm = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=3)
        print("Graph is ready")

        sess.run(init)
        for epoch in range(epochs):
            start_time = time.time()
            avg_cost = 0.
            for i in range(total_batch):
                batch_xs = train_images[i * batch_size:(i + 1) * batch_size]
                batch_ys = train_labels[i * batch_size:(i + 1) * batch_size]
                sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += sess.run(cross_entropy, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
                # Display logs per epoch step
            if epoch % display_step == 0:
                print("epoch:", '%04d' % (epoch), "cost_train=", "{:.9f}".format(avg_cost))
                print("time per epoch : %s" % (time.time() - start_time))
                print("validation accuracy : %s" , accuracy.eval(feed_dict={x: test_images, y: test_labels}),
                      "validation loss : %s", sess.run(cross_entropy, feed_dict={x: test_images, y: test_labels}))
            if epoch % save_step == 0:
                saver_path = saver.save(sess, "/home/user/JINGYU_KO/DEEPEST/WPI/VGG11/test_weight.ckpt-" + str(epoch))
                print("Saved!")
        print("Train finished!")

    def predict(self, images):
        """
        predict the class given the images
        :param images: the test set image
        :return: list with shape [None, 1], where None is sample size and 1 is class num integer
        """
        # implement here
        preds = None
        return preds