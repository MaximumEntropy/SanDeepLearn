# SanDeepLearn

# MNIST Multilayer Perceptron

```
train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(dataset='mnist')

network = SequentialNetwork(input_type='2d', output_type='multiple_class')
network.add(FullyConnectedLayer(train_x.shape[1], 500, activation='tanh'))
network.add(FullyConnectedLayer(500, 10, activation='tanh'))
network.add(SoftMaxLayer(hierarchical=False))

network.compile(loss='categorical_crossentropy')

network.train(train_x, train_y, nb_epochs=10, valid_x=dev_x, valid_y=dev_y, test_x=test_x, test_y=test_y)

```

# MNIST Le-Net 5

```
train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(dataset='mnist')

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

network.compile(loss='categorical_crossentropy', lr=0.1)

network.train(train_x, train_y, nb_epochs=10, valid_x=dev_x, valid_y=dev_y, test_x=test_x, test_y=test_y)

```

# Transmembrane helix prediction using Recurrent Neural Networks

train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(dataset='mnist')

# Recurrent Neural Network

network = RecurrentNetwork(input_type='1d', output_type='single_class', embedding=True)
network.add(EmbeddingLayer(20, 50, name='embedding'))
network.add(RNN(50, 50, name='rnn'))
network.add(FullyConnectedLayer(50, 1, name='fc'))
network.compile(lr=0.01)
network.train(train_x, train_y, batch_size='online', nb_epochs=10)

# Long Short-term Memory Network

network = RecurrentNetwork(input_type='1d', output_type='single_class', embedding=True)
network.add(EmbeddingLayer(20, 50, name='embedding'))
network.add(LSTM(50, 50, name='rnn'))
network.add(FullyConnectedLayer(50, 1, name='fc'))
network.compile(lr=0.01)
network.train(train_x, train_y, batch_size='online', nb_epochs=10)

# Bidirectional Recurrent Neural Network

network = RecurrentNetwork(input_type='1d', output_type='single_class', embedding=True)
network.add(EmbeddingLayer(20, 50, name='embedding'))
network.add(BiRNN(RNN(50, 50, name='forward_rnn'), RNN(50, 50, name='backward_rnn')))
network.add(FullyConnectedLayer(100, 1, name='fc'))
network.compile(lr=0.001)
network.train(train_x, train_y, batch_size='online', nb_epochs=10)

# Bidirectional Long Short-term Memory Network

network = RecurrentNetwork(input_type='1d', output_type='single_class', embedding=True)
network.add(EmbeddingLayer(20, 50, name='embedding'))
network.add(BiLSTM(LSTM(50, 50, name='forward_lstm'), LSTM(50, 50, name='backward_lstm')))
network.add(FullyConnectedLayer(100, 1, name='fc'))
network.compile(lr=0.001)
network.train(train_x, train_y, batch_size='online', nb_epochs=10)

```