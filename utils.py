import keras
import numpy as np
from collections import namedtuple

    
def get_mnist():
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = np.reshape(X_train, [X_train.shape[0], -1]).astype('float32') / 255.
    X_test  = np.reshape(X_test , [X_test.shape[0] , -1]).astype('float32') / 255.
    X_train = X_train * 2.0 - 1.0
    X_test  = X_test  * 2.0 - 1.0

    Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes).astype('float32')
    Y_test  = keras.utils.np_utils.to_categorical(y_test, nb_classes).astype('float32')

    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    trn = Dataset(X_train, Y_train, y_train, nb_classes)
    tst = Dataset(X_test , Y_test, y_test, nb_classes)

    del X_train, X_test, Y_train, Y_test, y_train, y_test
    
    return trn, tst