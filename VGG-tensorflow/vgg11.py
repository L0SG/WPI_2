import tensorflow as tf
import numpy as np

class vgg11:
    """
    the standard vgg11 model
    should contain images, labels, weights and session
    """
    def __init__(self, images, labels, weights=None, sess=None):
