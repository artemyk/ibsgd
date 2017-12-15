import keras
import keras.backend as K

import numpy as np

transpose = K.transpose
if K._BACKEND == 'tensorflow':
    import tensorflow as tf
    def tensor_eye(size):
        return tf.eye(size)
elif K._BACKEND == 'theano':
    import theano.tensor as T
    def tensor_eye(size):
        return T.eye(size)
elif K._BACKEND == 'cntk':
    raise Exception('Unsupported')
    import cntk as C
    def transpose(x):
        return C.swapaxes(x, -1, -2)
else:
    raise Exception('Unknown backend')


def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = K.expand_dims(K.sum(K.square(X), axis=1), 1)
    dists = x2 + transpose(x2) - 2*K.dot(X, transpose(X))
    return dists

def get_shape(x):
    dims = K.cast( K.shape(x)[1], K.floatx() ) 
    N    = K.cast( K.shape(x)[0], K.floatx() )
    return dims, N

def kde_entropy_from_dists_loo(dists, x, var):
    dims, N = get_shape(x)
    dists2 = dists + tensor_eye(K.cast(N, 'int32')) * 10e20
    dists2 = dists2 / (2*var)
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    lprobs  = K.logsumexp(-dists2, axis=1) - K.log(N-1) - normconst
    h = -K.mean(lprobs)
    return h

def entropy_estimator_kl(x, var):
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    lprobs = K.logsumexp(-dists2, axis=1) - K.log(N) - normconst
    h = -K.mean(lprobs)
    return dims/2 + h

def entropy_estimator_bd(x, var):
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats

    dims = output.shape[1]
    normconst = (dims/2.0)*(np.log(2*np.pi*var) + 1)
    return normconst

