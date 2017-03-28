import tensorflow as tf
import numpy as np

class vgg11:
    """
    the standard vgg11 model
    should contain images, labels, weights and session
    """
    def __init__(self, images, labels, weights=None, sess=None):
        """
        initialize & build the model
        :param images: image batch tensor for placeholder x
        :param labels: image labels tensor for placeholder y
        :param weights: if not none, load pre-computed weights
        :param sess: pass the session defined from the main script
        """
        self.images = images
        self.labels = labels
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
        :return: vgg11 model
        """
        # impelement here
        model = None
        return model


    def train(self, epochs, val_split, save_weights):
        """
        train the model with given epochs and val_split
        if save_weights == True, save the model weight for future use
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