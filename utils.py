#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy as np
import pickle, gzip

__author__ = "Sandeep Subramanian"
__maintainer__ = "Sandeep Subramanian"
__email__ = "sandeep.subramanian@gmail.com"

def get_data(dataset='mnist'):

	train, dev, test = pickle.load(gzip.open('data/mnist.pkl.gz', 'rb'))

	train_x, train_y = train[0], train[1].astype(np.int32)
	dev_x, dev_y = dev[0], dev[1].astype(np.int32)
	test_x, test_y = test[0], test[1].astype(np.int32)

	train_yy = np.zeros((train_y.shape[0], 10)).astype(np.int32)
	dev_yy = np.zeros((dev_y.shape[0], 10)).astype(np.int32)
	test_yy = np.zeros((test_y.shape[0], 10)).astype(np.int32)

	for ind, val in enumerate(train_y):
		train_yy[ind][val] = 1

	return train_x, train_yy, dev_x, dev_y, test_x, test_y

def get_weights(low, high, shape):

	"""
	Returns a weight matrix based on the activation function for that layer

	Initialization Strategy: http://deeplearning.net/tutorial/mlp.html

	"""

	weights = np.random.uniform(low=low, high=high, size=shape).astype(theano.config.floatX)
	return theano.shared(weights, borrow=True)

def get_bias(output_dim):

	"""
	Returns a bias vector for a layer initialized with zeros
	"""

	return theano.shared(np.zeros(output_dim, ).astype(theano.config.floatX), borrow=True)