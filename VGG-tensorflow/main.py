import tensorflow as tf
import numpy as np
from vgg import vgg11
import load_data

(train_x, train_y), (test_x, test_y) = load_data.load_data()
train_x=train_x[0:5000]
train_y=train_y[0:5000]
# if trained the model before, load the weights
weights = None

with tf.Session() as sess:
    # instantiate the model
    vgg_model = vgg11(weights=None, sess=sess)
    # train the model
    vgg_model.train(images=train_x, labels=train_y,
                    epochs=100, val_split=0.1, save_weights=True)
    # predict labels for the test images
    preds = vgg_model.predict(images=None)

    # calculate accuracy
    accuracy = np.sum([preds[i] == test_y[i] for i in xrange(test_y.shape[0])])
    accuracy /= test_y.shape[0]
    print('test accuracy: ' + str(accuracy))