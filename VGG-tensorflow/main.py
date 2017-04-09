import tensorflow as tf
from vgg import vgg11
import cifar100_utils
import yaml

with open("vgg.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

(train_x, train_y), (test_x, test_y) = cifar100_utils.load_data()
shape_x = train_x[0].shape
shape_y = train_y[0].shape
# if trained the model before, load the weights
weights = None
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    # instantiate the model
    vgg_model = vgg11(weights=None, sess=sess, shape_x = shape_x, shape_y = shape_y)
    # train the model
    vgg_model.train(images=train_x, labels=train_y, save_weights=False, config=config)
    # predict labels for the test images
    #preds = vgg_model.predict(images=None)

    # calculate accuracy
    #accuracy = np.sum([preds[i] == test_y[i] for i in xrange(test_y.shape[0])])
    #accuracy /= test_y.shape[0]
    #print('test accuracy: ' + str(accuracy))
