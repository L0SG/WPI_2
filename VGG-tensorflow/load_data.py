
from keras.datasets import cifar100
from keras.utils import to_categorical

def load_data(name=None):
    '''
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
    '''
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

    return (train_x, train_y), (test_x, test_y)