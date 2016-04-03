#!/usr/bin/env python

from optimizers import Optimizer
from layer import SoftMaxLayer, EmbeddingLayer
from utils import get_weights, get_bias

import theano
import theano.tensor as T
import numpy as np
theano.config.floatX = 'float32'

__author__ = "Sandeep Subramanian"
__version__ = "1.0"
__maintainer__ = "Sandeep Subramanian"
__email__ = "sandeep.subramanian@gmail.com"


class RecurrentNetwork:

	def __init__(self, input_type='2d', output_type='single_class', embedding=False):

		"""
		Constructor for a recurrent network
		"""

		# __TODO__ add 3d and 4d support
		if input_type == '1d' and embedding:
			self.input = T.ivector()
		if input_type == '1d' and not embedding:
			raise NotImplementedError("Cannot use 1D input without having an EmbeddingLayer")
		if input_type == '2d' and embedding:
			raise NotImplementedError("Cannot used 2D input with EmbeddingLayer")
		elif input_type == '2d' and not embedding:
			self.input = T.fmatrix()
		
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
		
		if not isinstance(layer_object, SoftMaxLayer) and (not (isinstance(layer_object, EmbeddingLayer) and layer_object.pretrained is not None)):
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
			loss = ((activation.T - self.output) ** 2).mean()
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
		Evaluates the current network
		"""
		# __TODO__ this evaluation works only for MNIST

		return self.f_eval(input)


	def train(self, train_x, train_y, dev_x=None, dev_y=None, test_x=None,  test_y=None, nb_epochs=20, batch_size=100):

		"""
		Train the model for the given input, number of epochs and batch size
		"""
		if not self.compiled:
			raise NotCompiledError("Network hasn't been compiled yet.")
			return

		if isinstance(batch_size, int):

			for epoch in xrange(nb_epochs):
				costs = []
				for batch in xrange(0, train_x.shape[0], batch_size):
					cost = self.f_train(
						train_x[batch:batch + batch_size], 
						train_y[batch:batch + batch_size]
					)
					costs.append(cost)

		elif batch_size == 'online':

			print 'Training in online mode ...'

			for epoch in xrange(nb_epochs):
				costs = []
				for data_point, labels in zip(train_x, train_y):
					cost = self.f_train(
						data_point,
						labels
					)
				costs.append(cost)
			
				print 'Epoch %d Training Loss : %f ' %(epoch, np.mean(costs))


class RNN:

	"""
	Recurrent Neural Network
	"""

	def __init__(self, input_dim, output_dim, activation='sigmoid', embedding=False, name='rnn', return_type='all'):

		# __TODO__ add deep recurrent network

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.return_type = return_type

		# Set the activation function for this layer
		if activation == 'sigmoid':
			self.activation = T.nnet.sigmoid
			low = -4 * np.sqrt(6. / (input_dim + output_dim))
			high = 4 * np.sqrt(6. / (input_dim + output_dim))

		elif activation == 'tanh':
			self.activation = T.tanh
			low = -1.0 * np.sqrt(6. / (input_dim + output_dim))
			high = np.sqrt(6. / (input_dim + output_dim))
		
		elif self.activation == 'linear':
			self.activation = None

		else:
			raise NotImplementedError("Unknown activation")

		self.weights = get_weights(shape=(input_dim, output_dim), name=name + '__weights')
		self.recurrent_weights = get_weights(shape=(output_dim, output_dim), name=name + '__recurrent_weights')
		
		self.bias = get_bias(output_dim, name=name + '__bias')
		self.recurrent_bias = get_bias(output_dim, name=name + '__recurrent_bias')

		self.h = None

		self.params = [self.weights, self.recurrent_weights, self.bias, self.recurrent_bias]

	def fprop(self, input):

		def recurrence_helper(current_input, recurrent_input):

			return self.activation(T.dot(current_input, self.weights) + T.dot(recurrent_input, self.recurrent_weights) + self.bias)

		outputs_info = self.recurrent_bias

		self.h, _ = theano.scan(
				fn=recurrence_helper,
				sequences=input,
				outputs_info=self.recurrent_bias,
				n_steps=input.shape[0]
			)

		if self.return_type == 'all':
			return self.h
		elif self.return_type == 'last':
			return self.h[-1]
		else:
			raise NotImplementedError("Unknown return type")

class LSTM:

	"""
	Long Short-term Memory Network
	"""

	def __init__(self, input_dim, output_dim, embedding=False, name='lstm', return_type='all'):

		# __TODO__ add deep LSTM

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.return_type = return_type

		low_sigmoid = -4 * np.sqrt(6. / (self.input_dim + self.output_dim))
		high_sigmoid = 4 * np.sqrt(6. / (self.input_dim + self.output_dim))

		low_tanh = -1.0 * np.sqrt(6. / (input_dim + output_dim))
		high_tanh = np.sqrt(6. / (input_dim + output_dim))
		
		# Intialize forget gate weights
		self.w_fx = get_weights(shape=(input_dim, output_dim), name=name + '__w_fx')
		self.w_fh = get_weights(shape=(output_dim, output_dim), name=name + '__w_fh')
		self.w_fc = get_weights(shape=(output_dim, output_dim), name=name + '__w_fc')

		# Intialize input gate weights
		self.w_ix = get_weights(shape=(input_dim, output_dim), name=name + '__w_ix')
		self.w_ih = get_weights(high=high_sigmoid, low=low_sigmoid, shape=(output_dim, output_dim), name=name + '__w_ih')
		self.w_ic = get_weights(high=high_sigmoid, low=low_sigmoid, shape=(output_dim, output_dim), name=name + '__w_ic')

		# Intialize output gate weights
		self.w_ox = get_weights(shape=(input_dim, output_dim), name=name + '__w_ox')
		self.w_oh = get_weights(shape=(output_dim, output_dim), name=name + '__w_oh')
		self.w_oc = get_weights(shape=(output_dim, output_dim), name=name + '__w_oc')

		# Initialize cell weights
		self.w_cx = get_weights(shape=(input_dim, output_dim), name=name + '__w_cx')
		self.w_ch = get_weights(shape=(output_dim, output_dim), name=name + '__w_ch')

		# Initialize bias for every gate
		self.b_f = get_bias(output_dim, name=name + '__b_f')
		self.b_i = get_bias(output_dim, name=name + '__b_i')
		self.b_o = get_bias(output_dim, name=name + '__b_o')
		self.b_c = get_bias(output_dim, name=name + '__b_c')
		
		self.c_0 = get_bias(output_dim, name=name + '__c_0')
		self.h_0 = get_bias(output_dim, name=name + '__h_0')

		self.h = None

		self.params = [self.w_fx, self.w_fh, self.w_fc, self.w_ix, self.w_ih, self.w_ic, self.w_ox, self.w_oh, self.w_oc, self.w_cx, self.w_ch, self.b_f, self.b_i, self.b_o, self.b_c, self.c_0, self.h_0]

	def fprop(self, input):

		def recurrence_helper(current_input, recurrent_input, cell_state):

			input_activation = T.nnet.sigmoid(T.dot(current_input, self.w_ix) + T.dot(recurrent_input, self.w_ih) + T.dot(cell_state, self.w_ic) + self.b_i)
			forget_activation = T.nnet.sigmoid(T.dot(current_input, self.w_fx) + T.dot(recurrent_input, self.w_fh) + T.dot(cell_state, self.w_fc) + self.b_f)
			cell_activation = forget_activation * cell_state + input_activation * T.tanh(T.dot(self.w_cx, current_input) + T.dot(self.w_ch, recurrent_input) + self.b_c)
			output_activation = T.nnet.sigmoid(T.dot(current_input, self.w_ox) + T.dot(recurrent_input, self.w_oh) + T.dot(cell_state, self.w_oc) + self.b_o)
			cell_output = output_activation * T.tanh(cell_activation)
			return [cell_state, cell_output]

		outputs_info = [self.c_0, self.h_0]
		[_, self.h], updates = theano.scan(
            fn=recurrence_helper,
            sequences=input,
            outputs_info=outputs_info,
            n_steps=input.shape[0]
        )

		if self.return_type == 'all':
			return self.h
		elif self.return_type == 'last':
			return self.h[-1]
		else:
			raise NotImplementedError("Unknown return type")

class FastLSTM:

	"""
	Faster Long Short-term Memory Network by using just 2 weight matrices
	"""

	def __init__(self, input_dim, output_dim, name='lstm', return_type='all'):

		# __TODO__ add deep LSTM

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.return_type = return_type

		low_sigmoid = -4 * np.sqrt(6. / (self.input_dim + self.output_dim))
		high_sigmoid = 4 * np.sqrt(6. / (self.input_dim + self.output_dim))
		
		self.W = get_weights(shape=(input_dim, output_dim * 4), name=name + '__W')
		self.U = get_weights(shape=(output_dim, output_dim * 4), name=name + '__U')
		self.b = get_bias(output_dim * 4, name=name + '__b')

		self.c_0 = get_bias(output_dim, name=name + '__c_0')
		self.h_0 = get_bias(output_dim, name=name + '__h_0')

		self.h = None

		self.params = [self.W, self.U, self.b, self.c_0, self.h_0]

	def _partition_weights(self, matrix, n):
		return matrix[:, n * self.output_dim: (n+1) * self.output_dim]

	def fprop(self, input):

		def recurrence_helper(current_input, cell_state, recurrent_input):
			recurrent_output = current_input + T.dot(recurrent_input, self.U)

			input_activation = T.nnet.sigmoid(self._partition_weights(recurrent_output, 0))
			forget_activation = T.nnet.sigmoid(self._partition_weights(recurrent_output, 1))
			output_activation = T.nnet.sigmoid(self._partition_weights(recurrent_output, 2))
			cell_activation = T.tanh(self._partition_weights(recurrent_output, 3))

			cell_output = forget_activation * cell_state + input_activation * cell_activation

			return [cell_output, output_activation * T.tanh(cell_output)]

		pre_activation = T.dot(input.dimshuffle(1, 0, 2), self.W) + self.b

		outputs_info = [T.alloc(x, input.shape[0], self.output_dim) for x in [self.c_0, self.h_0]]
        
		[_, self.h], updates = theano.scan(
            fn=recurrence_helper,
            sequences=pre_activation,
            outputs_info=outputs_info,
            n_steps=input.shape[1]
        )

		if self.return_type == 'all':
			return self.h
		elif self.return_type == 'last':
			return self.h[-1]
		else:
			raise NotImplementedError("Unknown return type")

class BiRNN:

	"""
	Bidirectional Recurrent Neural Network
	"""

	def __init__(self, forward_rnn, backward_rnn):

		self.forward_rnn = forward_rnn
		self.backward_rnn = backward_rnn

		self.params = self.forward_rnn.params + self.backward_rnn.params

	def fprop(self, input):

		assert self.forward_rnn.return_type == self.backward_rnn.return_type

		if self.forward_rnn.return_type == 'all':
			return T.concatenate((self.forward_rnn.fprop(input), self.backward_rnn.fprop(input[::-1])), axis=1)
		elif self.backward_rnn.return_type == 'last':
			return T.concatenate((self.forward_rnn.fprop(input), self.backward_rnn.fprop(input[::-1])))

class BiLSTM:

	"""
	Bidirectional Long Short-term Memory Network
	"""

	def __init__(self, forward_lstm, backward_lstm):

		self.forward_lstm = forward_lstm
		self.backward_lstm = backward_lstm

		self.params = self.forward_lstm.params + self.backward_lstm.params

	def fprop(self, input):

		assert self.forward_lstm.return_type == self.backward_lstm.return_type
		
		if self.forward_lstm.return_type == 'all':
			return T.concatenate((self.forward_lstm.fprop(input), self.backward_lstm.fprop(input[::-1])), axis=1)
		elif self.forward_lstm.return_type == 'last':
			return T.concatenate((self.forward_lstm.fprop(input), self.backward_lstm.fprop(input[::-1])))
