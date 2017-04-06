import tensorflow as tf
import numpy as np
from vgg import vgg11
import cifar100_utils


(train_x, train_y), (test_x, test_y) = cifar100_utils.load_data()
shape_x = train_x[0].shape
shape_y = train_y[0].shape
# if trained the model before, load the weights
weights = None
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # instantiate the model
    vgg_model = vgg11(weights=None, sess=sess, shape_x = shape_x, shape_y = shape_y)
    # train the model
    vgg_model.train(images=train_x, labels=train_y,
                    epochs=10, val_split=0.005, save_weights=True)
    # predict labels for the test images
    #preds = vgg_model.predict(images=None)

    # calculate accuracy
    #accuracy = np.sum([preds[i] == test_y[i] for i in xrange(test_y.shape[0])])
    #accuracy /= test_y.shape[0]
    #print('test accuracy: ' + str(accuracy))
