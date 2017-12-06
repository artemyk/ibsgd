# Simplified MI computation code from https://github.com/ravidziv/IDNNs
import numpy as np

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

def bin_calc_information(inputdata, layerdata, num_of_bins):
    p_xs, unique_inverse_x = get_unique_probs(inputdata)
    
    bins = np.linspace(-1, 1, num_of_bins, dtype='float32') 
    digitized = bins[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins) - 1].reshape(len(layerdata), -1)
    p_ts, _ = get_unique_probs( digitized )
    
    H_LAYER = -np.sum(p_ts * np.log(p_ts))
    H_LAYER_GIVEN_INPUT = 0.
    for xval in unique_inverse_x:
        p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == xval, :])
        H_LAYER_GIVEN_INPUT += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x))
    return H_LAYER - H_LAYER_GIVEN_INPUT
