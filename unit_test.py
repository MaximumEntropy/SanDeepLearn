from sequential import SequentialNetwork
from layer import FullyConnectedLayer, SoftMaxLayer, Convolution2DLayer, EmbeddingLayer
from utils import get_data
from recurrent import RecurrentNetwork, RNN, LSTM, BiRNN, BiLSTM

import numpy as np
import pickle, gzip

def unit_test_mlp(dropout_test=False, optimizer='sgd'):

	train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(dataset='mnist')

	network = SequentialNetwork(input_type='2d', output_type='multiple_class')
	network.add(FullyConnectedLayer(train_x.shape[1], 500, activation='tanh'))
	network.add(FullyConnectedLayer(500, 10, activation='tanh'))
	network.add(SoftMaxLayer(hierarchical=False))

	if optimizer == 'sgd':
		network.compile(loss='categorical_crossentropy', optimizer='sgd')

	elif optimizer == 'adagrad':
		network.compile(loss='categorical_crossentropy', optimizer='adagrad')

	elif optimizer == 'rmsprop':
		network.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	elif optimizer == 'adam':
		network.compile(loss='categorical_crossentropy', optimizer='adam')

	network.train(train_x, train_y, nb_epochs=10)

	print 'Accuracy on dev : %f ' %((np.argmax(network.predict(dev_x), axis=1) != dev_y).mean())
	print 'Accuracy on test : %f ' %((np.argmax(network.predict(test_x), axis=1) != test_y).mean())

def unit_test_conv(dropout_test=False, optimizer='sgd'):

	train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(dataset='mnist')

	train_x = train_x.reshape(train_x.shape[0], 1, int(np.sqrt(train_x.shape[1])), int(np.sqrt(train_x.shape[1])))
	dev_x = dev_x.reshape(dev_x.shape[0], 1, int(np.sqrt(dev_x.shape[1])), int(np.sqrt(dev_x.shape[1])))
	test_x = test_x.reshape(test_x.shape[0], 1, int(np.sqrt(test_x.shape[1])), int(np.sqrt(test_x.shape[1])))

	network = SequentialNetwork(input_type='4d', output_type='multiple_class')

	convolution_layer0 = Convolution2DLayer(
	    input_height=train_x.shape[2], 
	    input_width=train_x.shape[3], 
	    filter_width=5, 
	    filter_height=5, 
	    num_filters=20, 
	    num_feature_maps=1, 
	    flatten=False, 
	    wide=False
	)

	convolution_layer1 = Convolution2DLayer(
	    input_height=convolution_layer0.output_height_shape, 
	    input_width=convolution_layer0.output_width_shape, 
	    filter_width=5, 
	    filter_height=5, 
	    num_filters=50, 
	    num_feature_maps=20, 
	    flatten=True, 
	    wide=False
	)

	network.add(convolution_layer0)
	network.add(convolution_layer1)
	network.add(FullyConnectedLayer(800, 500, activation='tanh'))
	network.add(FullyConnectedLayer(500, 10, activation='tanh'))
	network.add(SoftMaxLayer(hierarchical=False))

	if optimizer == 'sgd':
		print 'Training with SGD ...'
		network.compile(loss='categorical_crossentropy', lr=0.1, optimizer='sgd')

	elif optimizer == 'adagrad':
		print 'Training with Adagrad ...'
		network.compile(loss='categorical_crossentropy', lr=0.01, optimizer='adagrad')

	elif optimizer == 'rmsprop':
		print 'Training with RMSprop ...'
		network.compile(loss='categorical_crossentropy', lr=0.001, optimizer='rmsprop')

	network.train(train_x, train_y, nb_epochs=10)

	print 'Accuracy on dev : %f ' %((np.argmax(network.predict(dev_x), axis=1) != dev_y).mean())
	print 'Accuracy on test : %f ' %((np.argmax(network.predict(test_x), axis=1) != test_y).mean())

def unit_test_rnn():

	train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(dataset='tmh')

	network = RecurrentNetwork(input_type='1d', output_type='single_class', embedding=True)
	network.add(EmbeddingLayer(20, 50, name='embedding'))
	network.add(RNN(50, 50, name='rnn'))
	network.add(FullyConnectedLayer(50, 1, name='fc'))
	network.compile(lr=0.01, optimizer='sgd')
	network.train(train_x, train_y, batch_size='online', nb_epochs=10)

def unit_test_birnn():
	
	train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(dataset='tmh')

	network = RecurrentNetwork(input_type='1d', output_type='single_class', embedding=True)
	network.add(EmbeddingLayer(20, 50, name='embedding'))
	network.add(BiRNN(RNN(50, 50, name='forward_rnn'), RNN(50, 50, name='backward_rnn')))
	network.add(FullyConnectedLayer(100, 1, name='fc'))
	network.compile(lr=0.001, optimizer='adagrad')
	network.train(train_x, train_y, batch_size='online', nb_epochs=10)

print 'Testing RNN ... '
unit_test_rnn()
print 'Testing BiRNN ... '
unit_test_birnn()
print 'Testing Multi-layer Perceptron ...'
unit_test_mlp(optimizer='adam')
print 'Testing Convolutional Neural Network ...'
unit_test_conv(optimizer='rmsprop')