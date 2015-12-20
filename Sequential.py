#!/usr/bin/env python

from Optimizers import Optimizer
from Layer import SoftMaxLayer

import theano
import theano.tensor as T
import numpy as np

__author__ = "Sandeep Subramanian"
__version__ = "1.0"
__maintainer__ = "Sandeep Subramanian"
__email__ = "sandeep.subramanian@gmail.com"

class SequentialNetwork:

	def __init__(self):

		"""
		Constructor for a sequential network
		"""

		self.input = T.fmatrix()
		self.output = T.imatrix()
		self.layers = []
		self.params = []
		self.compiled = False

	def add(self, layer_object):

		"""
		Add a layer to the network
		"""

		if len(self.layers) != 0 and not isinstance(layer_object, SoftMaxLayer):
			if layer_object.input_dim != self.layers[-1].output_dim:
				raise ValueError("Layer shape mismatch input dimension must be of size %d" %(self.layers[-1].output_dim))
		
		self.layers.append(layer_object)
		
		if not isinstance(layer_object, SoftMaxLayer):
			self.params.extend([layer_object.weights, layer_object.bias])

	def compile(self, loss='squared_error', optimizer='sgd'):

		"""
		Compile the network with a given loss function and optimization method
		"""

		# Propogate the input the input through the whole network

		prev_activation = self.input

		for ind, layer in enumerate(self.layers):

			activation = layer.fprop(prev_activation)
			prev_activation = activation

		# Compute loss

		if loss == 'squared_error':
			loss = ((activation - self.output) ** 2).mean()
		elif loss == 'categorical_crossentropy':
			loss = T.nnet.categorical_crossentropy(
				activation, 
				self.output
			).mean()

		# Select optimization strategy

		if optimizer == 'sgd':
			updates = Optimizer().sgd(loss, 
				self.params, 
				lr=0.01
			)
		
		else:
			raise NotImplementedError("Unknown optimization method")

		# Compile theano functions for training and evaluation

		self.f_train = theano.function(
    		inputs=[self.input, self.output],
    		outputs=loss,
    		updates=updates
		)

		self.f_eval = theano.function(
    		inputs=[self.input],
    		outputs=activation
		)

		self.compiled = True

	def evaluate(self, input_x, input_y):

		"""
		Evaluates the current network
		"""
		# __TODO__ this evaluation works only for MNIST

		return (np.argmax(self.f_eval(input_x), axis=1) != input_y).mean()


	def train(self, train_x, train_y, valid_x=None, valid_y=None, test_x=None,  test_y=None, nb_epochs=20, batch_size=100):

		"""
		Train the model for the given input, number of epochs and batch size
		"""
		if not self.compiled:
			raise NotCompiledError("Network hasn't been compiled yet.")
			return

		for epoch in xrange(nb_epochs):
			costs = []
			for batch in xrange(0, train_x.shape[0], batch_size):
				cost = self.f_train(
					train_x[batch:batch + batch_size], 
					train_y[batch:batch + batch_size]
				)
				costs.append(cost)
			
			print 'Epoch %d Training Loss : %f ' %(epoch, np.mean(costs))

			print 'Validation error : %f ' %(self.evaluate(valid_x, valid_y))
			print 'Test error : %f ' %(self.evaluate(test_x, test_y))

