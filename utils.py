import keras
import keras.backend as K
import numpy as np
from collections import namedtuple

def get_mnist2d():
    nb_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes).astype('float32')
    Y_test  = keras.utils.np_utils.to_categorical(y_test, nb_classes).astype('float32')
        
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    trn = Dataset(X_train, Y_train, y_train, nb_classes)
    tst = Dataset(X_test , Y_test, y_test, nb_classes)

    del X_train, X_test, Y_train, Y_test, y_train, y_test
    
    return trn, tst

def get_mnist():
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = np.reshape(X_train, [X_train.shape[0], -1]).astype('float32') / 255.
    X_test  = np.reshape(X_test , [X_test.shape[0] , -1]).astype('float32') / 255.
    #X_train = X_train * 2.0 - 1.0
    #X_test  = X_test  * 2.0 - 1.0

    Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes).astype('float32')
    Y_test  = keras.utils.np_utils.to_categorical(y_test, nb_classes).astype('float32')

    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    trn = Dataset(X_train, Y_train, y_train, nb_classes)
    tst = Dataset(X_test , Y_test, y_test, nb_classes)

    del X_train, X_test, Y_train, Y_test, y_train, y_test
    
    return trn, tst