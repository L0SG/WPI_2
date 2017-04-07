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

    def build_simple(self, images):
        self.conv3_32 = self.conv_layer(images,'con3_32',3,32)
        self.pool1 = self.max_pool(self.conv3_32, 'pool1')

        self.conv3_64 = self.condv_layer(self.pool1, 'conv3_64', 32, 64)
        self.pool2 = self.max_pool(self.conv3_64, 'pool2')
        shape = int(np.prod(self.pool2.get_shape()[1:]))

        self.fc1024 = self.fc_layer(self.pool2, 'fc1024', shape, 1024)
        self.fc100 = self.fc_layer(self.fc1024, 'fc100', 1024, 100)

        model = self.fc100

        return model

    def build(self, images):
        """
        build the model
        the model should output softmax class prediction tensor
        :return: vgg11 model
        """
        # implement here
        self.conv3_64 = self.conv_layer(images, 'con3_64', 3, 64)
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
        self.fc100 = self.fc_layer(self.fc4096_2, 'fc1000', 4096, 100)

        model = self.fc100

        return model

    def fc_bias(self, name, out_size):
        return tf.Variable(tf.constant(1.0, shape=[out_size], dtype='float32'), name=name)

    def fc_weight(self, name, in_size, out_size):
        return tf.Variable(tf.truncated_normal([in_size, out_size], dtype='float32', stddev=1e-1), name=name)

    def fc_layer(self, input, name, in_size, out_size):
        with tf.name_scope(name):
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
        with tf.name_scope(name):
            channel = self.conv_channel(name, pre_num_ch, num_ch)

            conv = tf.nn.conv2d(input, channel, [1, 1, 1, 1], padding='SAME')  # VAlID = without padding, SAME = with zero padding

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

        # implement here
        x = tf.placeholder("float", [None, 32, 32, 3])
        y = tf.placeholder("float", [None, 100])

        data_size = images.shape[0]
        train_data_size = (int)(data_size*(1-val_split))
        train_images, val_images = np.split(images, [train_data_size])
        train_labels, val_labels = np.split(labels, [train_data_size])

        batch_size = 25
        total_batch = train_data_size/batch_size
        display_step = 1
        #save_step = 100

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.build_simple(x)), name="loss")
        optm = tf.train.AdamOptimizer(1e-4).minimize(loss)
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.build_simple(x), 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        writer = tf.summary.FileWriter("./tmp/1")
        writer.add_graph(self.sess.graph)

        print("Graph is ready")

        self.sess.run(init)
        for epoch in range(epochs):
            start_time = time.time()
            avg_cost = 0.
            # batch randomization should be added.
            for i in range(total_batch):
                batch_xs = train_images[i * batch_size:(i + 1) * batch_size]
                batch_ys = train_labels[i * batch_size:(i + 1) * batch_size]
                self.sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += self.sess.run(loss, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

                # Display logs per epoch step
            if epoch % display_step == 0:
                print("epoch : %d, loss : %.5f" %(epoch,avg_cost))
                print("validation accuracy : %.5f, validation loss : %.5f"
                      %(self.sess.run(accuracy, feed_dict={x: val_images, y: val_labels}), self.sess.run(loss, feed_dict={x: val_images, y: val_labels})))
            print("Train finished!")


#            if epoch % save_step == 0:
#                saver_path = saver.save(self.sess, "test_weight.ckpt-" + str(epoch))
#                print("Saved!")


'''
    def predict(self, images):
        """
        predict the class given the images
        :param images: the test set image
        :return: list with shape [None, 1], where None is sample size and 1 is class num integer
        """
        # implement here
        N=size(images,0)
        
        mod = self.model
        
        predicted = np.zeros((N,10),dtype=np.int)
        preds = zeros((N,),dtype=np.int)
        image_tensor = tf.placeholder(tf.float32,[1,32,32,3])
        for i in range(N):
            image_tensor = images[i,:,:,:]
            predicted[i,:] = sess.run(self.model, feed_dict={ image_tensor : image_tensor})  
            preds[i,1] = tf.arg_max(predicted[i,:],1) 
        return preds
'''
