from __future__ import print_function
import numpy as np

# Original MI computation code from https://github.com/ravidziv/IDNNs

def extract_probs(label, x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
    pys = np.sum(label, axis=0) / float(label.shape[0])
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse_x, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]
    b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
    pxs = unique_counts / float(np.sum(unique_counts))
    p_y_given_x = []
    for i in range(0, len(unique_array)):
        indexs = unique_inverse_x == i
        py_x_current = np.mean(label[indexs, :], axis=0)
        p_y_given_x.append(py_x_current)
    p_y_given_x = np.array(p_y_given_x).T
    b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))
    return pys, pys1, p_y_given_x, b, unique_a, unique_inverse_x, unique_inverse_y, pxs

def calc_information_sampling(data, bins, pys1, pxs, label, b, p_YgX, unique_inverse_x,  unique_inverse_y):
    bins = bins.astype(np.float32)
    num_of_bins = bins.shape[0]
    # bins = stats.mstats.mquantiles(np.squeeze(data.reshape(1, -1)), np.linspace(0,1, num=num_of_bins))
    # hist, bin_edges = np.histogram(np.squeeze(data.reshape(1, -1)), normed=True)
    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
    local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y,
                                                     unique_array)
    return local_IXT, local_ITY

def calc_condtion_entropy(px, t_data, unique_inverse_x):
    # Condition entropy of t given x
    H2X_array = np.array(list(
        calc_entropy_for_specipic_t(t_data[unique_inverse_x == i, :], px[i])
                                   for i in range(px.shape[0])))
    H2X = np.sum(H2X_array)
    return H2X


def calc_information_from_mat(px, py, ps2, data, unique_inverse_x, unique_inverse_y, unique_array):
    """Calculate the MI based on binning of the data"""
    H2 = -np.sum(ps2 * np.log(ps2))
    H2X = calc_condtion_entropy(px, data, unique_inverse_x)
    H2Y = calc_condtion_entropy(py.T, data, unique_inverse_y)
    IY = H2 - H2Y
    IX = H2 - H2X
    return IX, IY

def calc_entropy_for_specipic_t(current_ts, px_i):
    """Calc entropy for specipic t"""
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_current_ts = unique_counts / float(sum(unique_counts))
    p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
    H2X = px_i * (-np.sum(p_current_ts * np.log(p_current_ts)))
    return H2X

if __name__ == "__main__":
    import simplebinmi, utils
    trn, tst = utils.get_mnist()
    
    import keras
    import keras.backend as K
    input_layer  = keras.layers.Input((trn.X.shape[1],))
    hidden_output = keras.layers.Dense(1024, activation='tanh')(input_layer)
    hidden_output = keras.layers.Dense(5  , activation='tanh')(hidden_output)
    hidden_output = keras.layers.Dense(5  , activation='tanh')(hidden_output)

    outputs  = keras.layers.Dense(trn.nb_classes, activation='softmax')(hidden_output)
    model = keras.models.Model(inputs=input_layer, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    inputs = model.inputs + [ K.learning_phase(),] 
    test_num_of_bins = 10
    bins = np.linspace(-1, 1, test_num_of_bins)

    for lndx, l in enumerate(model.layers[2:-1]):
        inputdata = trn.X[::20]
        label = trn.Y[::20]
        data = K.function(inputs, [l.output])([inputdata, 0])[0]
        pys, pys1, p_YgX, b, unique_a, unique_inverse_x, unique_inverse_y, pxs = extract_probs(label, inputdata)

        local_IXT, local_ITY = calc_information_sampling(data, bins, pys1, pxs, label, b, p_YgX, unique_inverse_x,
                                  unique_inverse_y)

        print('Original', local_IXT, local_ITY)
        mi = simplebinmi.bin_calc_information(inputdata, data, test_num_of_bins)
        print('Mine', mi, mi==local_IXT)
