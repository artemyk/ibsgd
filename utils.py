import keras
import keras.backend as K
import numpy as np
import scipy.io as sio
from pathlib2 import Path
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

def get_IB_data(ID):
    # Returns two namedtuples, with IB training and testing data
    #   trn.X is training data
    #   trn.y is trainiing class, with numbers from 0 to 9
    #   trn.Y is training class, but coded as a 10-dim vector with one entry set to 1
    # similarly for tst
    nb_classes = 2
    data_file = Path('datasets/IB_data_'+str(ID)+'.npz')
    if data_file.is_file():
        data = np.load('datasets/IB_data_'+str(ID)+'.npz')
    else:
        create_IB_data(ID)
        data = np.load('datasets/IB_data_'+str(ID)+'.npz')
        
    (X_train, y_train), (X_test, y_test) = (data['X_train'], data['y_train']), (data['X_test'], data['y_test'])

    Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes).astype('float32')
    Y_test  = keras.utils.np_utils.to_categorical(y_test, nb_classes).astype('float32')

    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    trn = Dataset(X_train, Y_train, y_train, nb_classes)
    tst = Dataset(X_test , Y_test, y_test, nb_classes)
    del X_train, X_test, Y_train, Y_test, y_train, y_test
    return trn, tst

def create_IB_data(idx):
    data_sets_org = load_data()
    data_sets = data_shuffle(data_sets_org, 80, shuffle_data=True)
    X_train, y_train, X_test, y_test = data_sets.train.data, data_sets.train.labels[:,0], data_sets.test.data, data_sets.test.labels[:,0]
    np.savez_compressed('datasets/IB_data_'+str(idx), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

def construct_full_dataset(trn, tst):
    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    X = np.concatenate((trn.X,tst.X))
    y = np.concatenate((trn.y,tst.y))
    Y = np.concatenate((trn.Y,tst.Y))
    return Dataset(X, Y, y, trn.nb_classes)
 
def load_data():
    """Load the data
    name - the name of the dataset
    return object with data and labels"""
    print ('Loading Data...')
    C = type('type_C', (object,), {})
    data_sets = C()
    d = sio.loadmat('datasets/var_u.mat')
    F = d['F']
    y = d['y']
    C = type('type_C', (object,), {})
    data_sets = C()
    data_sets.data = F
    data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    return data_sets

def shuffle_in_unison_inplace(a, b):
    """Shuffle the arrays randomly"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
    """Divided the data to train and test and shuffle it"""
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    C = type('type_C', (object,), {})
    data_sets = C()
    stop_train_index = perc(percent_of_train, data_sets_org.data.shape[0])
    start_test_index = stop_train_index
    
    if percent_of_train > min_test_data:
        start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
    else:
        shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.train.labels = shuffled_labels[:stop_train_index, :]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    data_sets.test.labels = shuffled_labels[start_test_index:, :]
    return data_sets
