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
            '''
            self.weights = {
                'wc1': tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1)),
                'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
                'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)),
                'wc4': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1)),
                'wc5': tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1)),
                'wc6': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'wc7': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'wc8': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
                'wd1': tf.Variable(tf.truncated_normal([512, 512], stddev=0.1)),
                'wd2': tf.Variable(tf.truncated_normal([512, 512], stddev=0.1)),
                'wd3': tf.Variable(tf.truncated_normal([512, 200], stddev=0.1)),
                'wd4': tf.Variable(tf.truncated_normal([200, 100], stddev=0.1)),
                'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
                'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
                'bc3': tf.Variable(tf.random_normal([256], stddev=0.1)),
                'bc4': tf.Variable(tf.random_normal([256], stddev=0.1)),
                'bc5': tf.Variable(tf.random_normal([512], stddev=0.1)),
                'bc6': tf.Variable(tf.random_normal([512], stddev=0.1)),
                'bc7': tf.Variable(tf.random_normal([512], stddev=0.1)),
                'bc8': tf.Variable(tf.random_normal([512], stddev=0.1)),
                'bd1': tf.Variable(tf.random_normal([512], stddev=0.1)),
                'bd2': tf.Variable(tf.random_normal([512], stddev=0.1)),
                'bd3': tf.Variable(tf.random_normal([200], stddev=0.1)),
                'bd4': tf.Variable(tf.random_normal([100], stddev=0.1))
            }
            '''
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

        # CONV LAYER 1
        conv1=self.conv_layer(self.x, 3, 64, "conv1", isPooling=True)
        # CONV LAYER 2
        conv2=self.conv_layer(conv1, 64, 128, "conv2", isPooling=True)
        # CONV LAYER 3 & 4
        conv3=self.conv_layer(conv2, 128, 256, "conv3", isPooling=False)
        conv4=self.conv_layer(conv3, 256, 256, "conv4", isPooling=True)
        # CONV LAYER 5 & 6
        conv5=self.conv_layer(conv4, 256, 512, "conv5", isPooling=False)
        conv6=self.conv_layer(conv5, 512, 512, "conv6", isPooling=True)
        # CONV LAYER 7 & 8
        conv7=self.conv_layer(conv6, 512, 512, "conv7", isPooling=False)
        conv8=self.conv_layer(conv7, 512, 512, "conv8", isPooling=True)
        # VECTORIZE
        print(conv8.get_shape())
        flat = tf.contrib.layers.flatten(conv8)
        print(flat.get_shape())
        # FULLY CONNECTED LAYER 1
        fc1=self.fc_layer(flat, 512, 512, "dense1", isDropout=True)
        # FULLY CONNECTED LAYER 2
        fc2=self.fc_layer(fc1, 512, 200, "dense2", isDropout=True)
        # FULLY CONNECTED LAYER 3
        fc3=self.fc_layer(fc2, 200, 100, "dense3", isDropout=False)
        # RETURN
        print ("CNN READY")
        return fc3

    def conv_layer(self, input, channels_in, channels_out, name, isPooling=False):
        with tf.name_scope(name):
            w = tf.Variable(tf.zeros([3, 3, channels_in, channels_out]), dtype=tf.float32,  name="w")
            b = tf.Variable(tf.zeros([channels_out]), dtype=tf.float32,  name="b")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME", name="conv") + b
            activ = tf.nn.relu(conv, name="activ")
            if(not isPooling) :
                return activ
            else :
                pool = tf.nn.max_pool(activ, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool")
                return pool

    def fc_layer(self, input, channels_in, channels_out, name, isDropout=False):
        with tf.name_scope(name):
            w = tf.Variable(tf.zeros([channels_in, channels_out]), dtype=tf.float32, name="w")
            b = tf.Variable(tf.zeros([channels_out]), dtype=tf.float32,  name="b")
            print(input.get_shape())
            print(w.get_shape())
            activ = tf.nn.relu(tf.matmul(input, w) + b, name="activ")
            if(not isDropout) :
                return activ
            else :
                drop = tf.nn.dropout(activ, 0.5)
                return drop


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
        train_data_size = (int)(data_size*(1-val_split))
        test_data_size = (int)(data_size-data_size*(1-val_split))
        train_images, test_images=np.split(images, [train_data_size])
        train_labels, test_labels=np.split(labels, [train_data_size])

        x = self.x
        y = self.y
        sess = self.sess

        batch_size = 25
        total_batch = (int)(train_data_size/batch_size)
        display_step = 10
        save_step = 100

        # impelement here
        pred = self.model
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred), name="loss")
        optm = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=3)

        writer = tf.summary.FileWriter("./tmp/1")
        writer.add_graph(sess.graph)

        print("Graph is ready")

        sess.run(init)
        for epoch in range(epochs):
            start_time = time.time()
            avg_cost = 0.
            for i in range(total_batch):
                batch_xs = train_images[i * batch_size:(i + 1) * batch_size]
                batch_ys = train_labels[i * batch_size:(i + 1) * batch_size]
                sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

                # Display logs per epoch step
            if epoch % display_step == 0:
                print("time per epoch : %s" % (time.time() - start_time),
                      "epoch:", '%04d' % (epoch), "cost_train=", "{:.9f}".format(avg_cost))
                print("validation accuracy : %s" , accuracy.eval(feed_dict={x: test_images, y: test_labels}),
                      "validation loss : %s", sess.run(loss, feed_dict={x: test_images, y: test_labels}))
            if epoch % save_step == 0:
#                saver_path = saver.save(sess, "/home/user/JINGYU_KO/DEEPEST/WPI/VGG11/test_weight.ckpt-" + str(epoch))
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