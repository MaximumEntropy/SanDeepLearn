"""Utility functions."""
import theano
import numpy as np
import pickle
import gzip
import scipy.io as sio


def get_data(dataset='mnist'):
    """Fetch dataset."""
    if dataset == 'mnist':

        train, dev, test = pickle.load(gzip.open('/u/subramas/Research/SanDeepLearn/data/mnist.pkl.gz', 'rb'))

        train_x, train_y = train[0], train[1].astype(np.int32)
        dev_x, dev_y = dev[0], dev[1].astype(np.int32)
        test_x, test_y = test[0], test[1].astype(np.int32)

        train_yy = np.zeros((train_y.shape[0], 10)).astype(np.int32)

        for ind, val in enumerate(train_y):
            train_yy[ind][val] = 1

        return train_x, train_yy, dev_x, dev_y, test_x, test_y

    elif dataset == 'tmh':

        dataset_train = sio.loadmat('data/train_dev_test_full_seq.mat')

        train_x = np.array([
            x.squeeze().astype(np.int32)
            for x in dataset_train['train_x'].squeeze()
        ])
        train_y = np.array([
            x.squeeze().astype(np.int32)
            for x in dataset_train['train_y'].squeeze()
        ])
        dev_x = np.array([
            x.squeeze().astype(np.int32)
            for x in dataset_train['dev_x'].squeeze()
        ])
        dev_y = np.array([
            x.squeeze().astype(np.int32)
            for x in dataset_train['dev_y'].squeeze()
        ])
        test_x = np.array([
            x.squeeze().astype(np.int32)
            for x in dataset_train['test_x'].squeeze()
        ])
        test_y = np.array([
            x.squeeze().astype(np.int32)
            for x in dataset_train['test_y'].squeeze()
        ])

        return train_x, train_y, dev_x, dev_y, test_x, test_y


def get_weights(shape, name, strategy='glorot'):
    """Return a randomly intialized weight matrix."""
    # Initialization Strategy: http://deeplearning.net/tutorial/mlp.html

    if strategy == 'glorot':
        drange = np.sqrt(6. / (np.sum(shape)))
        weights = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
        return theano.shared(
            np.array(weights).astype(theano.config.floatX),
            borrow=True,
            name=name
        )
    elif strategy == 'he2015':
        if len(shape) == 4:
            fan_in = np.prod(shape[1:])
        elif len(shape) == 2:
            fan_in = shape[0]
        weights = np.random.normal(
            loc=0,
            scale=np.sqrt(2.0 / fan_in),
            size=shape
        ).astype(theano.config.floatX)
        return theano.shared(weights, borrow=True, name=name)
    elif strategy == 'orthogonal':
        assert shape[0] == shape[1]
        W = np.random.randn(shape[0], shape[1])
        u, s, v = np.linalg.svd(W)
        return u.astype(theano.config.floatX)


def get_relu_weights(shape, name):
    """Return a weight matrix for the ReLU activation function."""
    # Initialization Strategy: http://arxiv.org/pdf/1502.01852v1.pdf

    weights = np.random.normal(
        loc=0,
        scale=np.sqrt(2.0 / (shape[0] + shape[1])),
        size=shape
    ).astype(theano.config.floatX)
    return theano.shared(weights, borrow=True, name=name)


def get_bias(output_dim, name):
    """Return a bias vector for a layer initialized with zeros."""
    return theano.shared(
        np.zeros(output_dim, ).astype(theano.config.floatX),
        borrow=True,
        name=name
    )


def get_highway_bias(output_dim, name):
    """Return a bias vector specific to highway networks."""
    # The vector is initialized with negative values.
    # Reference - http://arxiv.org/pdf/1505.00387v2.pdf
    return theano.shared(
        (np.zeros(output_dim, ) - 3).astype(theano.config.floatX),
        borrow=True,
        name=name
    )


def create_shared(numpy_array, name):
    """Create theano shared variable."""
    return theano.shared(
        numpy_array,
        borrow=True,
        name=name
    )


def zero_vector(length):
    """Zero vector for bias."""
    return np.zeros((length, )).astype('float32')


def ortho_weight(ndim):
    """Orthogonal weights."""
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """Standard normal weights."""
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')


def uniform_weight(nin, nout, scale=None):
    """Uniform weights."""
    if scale is None:
        scale = np.sqrt(6. / (nin + nout))

    W = np.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')
