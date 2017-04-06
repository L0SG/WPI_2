import numpy as np
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.utils import to_categorical

def load_data(name='cifar-100'):
    """
    load dataset
    :return: (train_x, train_y), (test_x, test_y) in numpy format
    """
    # If user didn't put the name of dataset, receive it.
    if name==None:
        print('Available data sets : cifar-10, cifar-100, mnist')
        name = input('Type the name of data set : ')


    elif name == 'cifar-10':
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()
        train_x = train_x.astype('float32') / 255.
        test_x = test_x.astype('float32') / 255.
        # Convert class vectors to binary class matrices.
        train_y = to_categorical(train_y, 10)
        test_y = to_categorical(test_y, 10)
        print("train_x.shape = " + str(train_x.shape))
        print("train_y.shape = " + str(train_y.shape))
        print("test_x.shape = " + str(test_x.shape))
        print("test_y.shape = " + str(test_y.shape))

    elif name == 'cifar-100':
        (train_x, train_y), (test_x, test_y) = cifar100.load_data()
        train_x = train_x.astype('float32') / 255.
        test_x = test_x.astype('float32') / 255.
        # Convert class vectors to binary class matrices.
        train_y = to_categorical(train_y, 100)
        test_y = to_categorical(test_y, 100)
        print("train_x.shape = " + str(train_x.shape))
        print("train_y.shape = " + str(train_y.shape))
        print("test_x.shape = " + str(test_x.shape))
        print("test_y.shape = " + str(test_y.shape))

    elif name == 'mnist':
        print("mnist dataset is loaded...")
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        train_x = train_x.astype('float32') / 255.
        test_x = test_x.astype('float32') / 255.
        train_x = train_x.reshape((len(train_x), np.prod(train_x.shape[1:])))
        test_x = test_x.reshape((len(test_x), np.prod(test_x.shape[1:])))
        # Convert class vectors to binary class matrices.
        train_y = to_categorical(train_y, 10)
        test_y = to_categorical(test_y, 10)
        print("train_x.shape = " + str(train_x.shape))
        print("train_y.shape = " + str(train_y.shape))
        print("test_x.shape = " + str(test_x.shape))
        print("test_y.shape = " + str(test_y.shape))

    else:
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        raise ValueError('%s is not a valid name of dataset' % (name))
        
    return (train_x, train_y), (test_x, test_y)