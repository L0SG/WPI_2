import tensorflow as tf
import numpy as np
from vgg import vgg11
import cifar100_utils

(train_x, train_y), (test_x, test_y) = cifar100_utils.load_data()

# if trained the model before, load the weights
weights = None

with tf.Session() as sess:
    # instantiate the model
    vgg_model = vgg11(images=train_x, labels=train_y, weights=None, sess=sess)
    # train the model
    vgg_model.train(epochs=100, val_split=0.1, save_weights=True)
    # predict labels for the test images
    preds = vgg_model.predict(images=None)

    # calculate accuracy
    accuracy = np.sum([preds[i] == test_y[i] for i in xrange(test_y.shape[0])])
    accuracy /= test_y.shape[0]
    print('test accuracy: ' + str(accuracy))
