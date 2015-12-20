#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy as np

__author__ = "Sandeep Subramanian"
__maintainer__ = "Sandeep Subramanian"
__email__ = "sandeep.subramanian@gmail.com"

def get_weights(activation, input_dim, output_dim):

	"""
	Returns a weight matrix based on the activation function for that layer

	Initialization Strategy: http://deeplearning.net/tutorial/mlp.html

	"""

	if activation == 'sigmoid':

		low = -4 * np.sqrt(6. / (input_dim + output_dim))
		high = 4 * np.sqrt(6. / (input_dim + output_dim))
		weights = np.random.uniform(low=low, high=high, size=(input_dim, output_dim)).astype(theano.config.floatX)
		return theano.shared(weights, borrow=True)

	elif activation == 'tanh':

		low = -1 * np.sqrt(6. / (input_dim + output_dim))
		high = np.sqrt(6. / (input_dim + output_dim))
		weights = np.random.uniform(low=low, high=high, size=(input_dim, output_dim)).astype(theano.config.floatX)
		return theano.shared(weights, borrow=True)

def get_bias(output_dim):

	"""
	Returns a bias vector for a layer initialized with zeros
	"""

	return theano.shared(np.zeros(output_dim, ).astype(theano.config.floatX), borrow=True)