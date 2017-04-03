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


    def build(self):
        """
        build the model
        the model should output softmax class prediction tensor
        :return: vgg11 model
        """
        # impelement here
        model = None
        return model


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
