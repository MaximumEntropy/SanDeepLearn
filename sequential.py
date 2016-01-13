#!/usr/bin/env python

from optimizers import Optimizer
from layer import SoftMaxLayer

import theano
import theano.tensor as T
import numpy as np
theano.config.floatX = 'float32'

__author__ = "Sandeep Subramanian"
__version__ = "1.0"
__maintainer__ = "Sandeep Subramanian"
__email__ = "sandeep.subramanian@gmail.com"

class SequentialNetwork:

	def __init__(self, input_type='2d', output_type='single_class'):

		"""
		Constructor for a sequential network
		"""

		if input_type == '2d' :
			self.input = T.fmatrix()
		elif input_type == '3d':
			self.input = T.tensor3()
		elif input_type == '4d':
			self.input = T.tensor4()

		if output_type == 'single_class':
			self.output = T.ivector()
		elif output_type == 'multiple_class':
			self.output = T.imatrix()
		elif output_type == 'regression':
			self.output = T.fvector()

		self.layers = []
		self.params = []
		self.compiled = False

	def add(self, layer_object):

		"""
		Add a layer to the network
		"""
		
		self.layers.append(layer_object)
		
		if not isinstance(layer_object, SoftMaxLayer):
			self.params.extend(layer_object.params)

	def compile(self, loss='squared_error', optimizer='sgd', lr=0.01):

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
			updates = Optimizer().sgd(
				loss, 
				self.params, 
				lr=lr
			)

		elif optimizer == 'adagrad':
			updates = Optimizer().adagrad(
				loss,
				self.params,
				lr=lr
			)

		elif optimizer == 'rmsprop':
			updates = Optimizer().rmsprop(
				loss,
				self.params,
				lr=lr
			)

		elif optimizer == 'adam':
			updates = Optimizer().adam(
				loss,
				self.params,
				lr=lr
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


	def predict(self, input):

		"""
		Returns a prediction for a given input
		"""
		
		return self.f_eval(input)


	def train(self, train_x, train_y, nb_epochs=20, batch_size=100):

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