#!/usr/bin/env python

from Utils import get_weights, get_bias

import theano
import theano.tensor as T
import numpy as np

__author__ = "Sandeep Subramanian"
__maintainer__ = "Sandeep Subramanian"
__email__ = "sandeep.subramanian@gmail.com"

class FullyConnectedLayer:

	def __init__(self, input_dim, output_dim, activation='sigmoid', name='fully_connected_layer'):

		"""
		Fully Connected Layer
		"""

		# Set input and output dimensions
		self.input_dim = input_dim
		self.output_dim = output_dim
		
		# Initialize weights & biases for this layer
		self.weights = get_weights(activation, input_dim, output_dim)
		self.bias = get_bias(output_dim)

		# Set the activation function for this layer
		if activation == 'sigmoid':
			self.activation = T.nnet.sigmoid
		elif activation == 'tanh':
			self.activation = T.tanh
		elif activation == 'softmax':
			self.activation = T.nnet.softmax
		elif self.activation == 'linear':
			self.activation = None
		else:
			raise NotImplementedError("Unknown activation")

	def fprop(self, input):

		# Propogate the input through the layer
		linear_activation = T.dot(input, self.weights) + self.bias
		if self.activation:
			return linear_activation
		else:
			return self.activation(linear_activation)

class SoftMaxLayer:

	"""
	Softmax Layer
	"""

	def __init__(self, hierarchical=False):

		# __TODO__ Add Hierarchical Softmax

		self.hierarchical = hierarchical

	def fprop(self, input):

		if not self.hierarchical:
			return T.nnet.softmax(input)
