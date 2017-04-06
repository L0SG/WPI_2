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
        # implement here
        self.x = tf.placeholder(tf.float32, shape=(None,32,32,3), name='image')
        self.y = tf.placeholder(tf.float32, shape=(None,100), name='label')

        self.conv3_64 = self.conv_layer(self.x, 'con3_64', 3, 64)
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
        self.fc4096_1 = self.fc_layer(self.pool5, 'fc4096_1', shape, 4096)
        self.fc4096_2 = self.fc_layer(self.fc4096_1, 'fc4096_2', 4096, 4096)
        self.fc100 = self.fc_layer(self.fc4096_2, 'fc100', 4096, 100)

        model = tf.nn.softmax(self.fc100, name='softmax_out')

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

            return tf.nn.relu(fc)

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
        #1 data split
        x = self.x
        y = self.y
        sess = self.sess

        data_size=images.shape[0]
        train_data_size = (int)(data_size*(1-val_split))
        test_data_size = (int)(data_size-data_size*(1-val_split))
        train_images, valid_images=np.split(images, [train_data_size])
        train_labels, valid_labels=np.split(labels, [train_data_size])
        valid_feed = {x: valid_images, y: valid_labels}

        batch_size = 25
        total_batch = (int)(train_data_size/batch_size)
        display_step = 1
        save_step = 100

        # impelement here
        pred = self.model
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred), name="loss")
            train_loss_summary = tf.summary.scalar('train_loss', loss)
            valid_loss_summary = tf.summary.scalar('valid_loss', loss)
        optm = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
            accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        valid_merged = tf.summary.merge([valid_loss_summary, accuracy_summary])
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
                train_feed = {x: batch_xs, y: batch_ys}
                train_loss, _ = sess.run([train_loss_summary, optm], train_feed)
                writer.add_summary(train_loss, (epoch * total_batch + i))

                # Display logs per epoch step
            if epoch % display_step == 0:
                valid_loss_acc = sess.run(valid_merged, valid_feed)
                writer.add_summary(valid_loss_acc, (epoch * total_batch))
                print("\r\b\rtime per epoch : %.2f" % (time.time() - start_time),
                      "epoch:", '%d' % (epoch), "cost_train=", "{:.9f}".format(avg_cost),
                      "\nvalidation accuracy : %s" , accuracy.eval(valid_feed),
                      "validation loss : %s", loss.eval(valid_feed), end='')
            if epoch % save_step == 0:
                saver.save(sess, "./test_weight.ckpt-" + str(epoch), (epoch * total_batch))
        print("\nTrain finished!")

    def predict(self, images):
        """
        predict the class given the images
        :param images: the test set image
        :return: list with shape [None, 1], where None is sample size and 1 is class num integer
        """
        # implement here
        N=np.size(images,0)
        
        mod = self.model
        
        predicted = np.zeros((N,10),dtype=np.int)
        preds = np.zeros((N,),dtype=np.int)
        image_tensor = tf.placeholder(tf.float32,[1,32,32,3])
        for i in range(N):
            image_tensor = images[i,:,:,:]
            predicted[i,:] = self.sess.run(self.model, feed_dict={ image_tensor : image_tensor})
            preds[i,1] = tf.arg_max(predicted[i,:],1) 
        return preds
