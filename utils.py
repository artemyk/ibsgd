import keras
import keras.backend as K
import numpy as np
from collections import namedtuple

def get_mnist():
    # Returns two namedtuples, with MNIST training and testing data
    #   trn.X is training data
    #   trn.y is trainiing class, with numbers from 0 to 9
    #   trn.Y is training class, but coded as a 10-dim vector with one entry set to 1
    # similarly for tst
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