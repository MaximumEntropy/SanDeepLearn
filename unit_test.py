"""Unit tests for RNNs and feedforward layers."""
from convolution import Convolution2DLayer, KMaxPoolingLayer
from layer import EmbeddingLayer, DropoutLayer, FullyConnectedLayer
from utils import get_data
from recurrent import RNN, LSTM
import theano
import theano.tensor as T
from optimizers import Optimizer
# from recurrent import BiLSTM, FastLSTM, MiLSTM

import numpy as np
import argparse
parser = argparse.ArgumentParser(
    description='Unit tests for RNNs and feedforward layers.'
)
parser.add_argument(
    '--layer',
    help='Layer to be tested.',
    required=True
)
parser.add_argument(
    '--optimizer',
    help='Optimization technique to be used.',
    default='SGD'
)
args = parser.parse_args()

layer = args.layer
optimization_method = args.optimizer

fclayers = [
    'fc',
    'cnn',
]

rnnlayers = [
    'rnn',
    'lstm',
]

misc = [
    'drp'  # dropout
]


if layer not in fclayers + rnnlayers + misc:
    raise ValueError(
        "Layer name not recognized. Implemented layers : %s"
        % (','.join(fclayers + rnnlayers))
    )

if layer in fclayers or layer in misc:
    print 'Fetching MNIST data ...'
    train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(dataset='mnist')
else:
    print 'Fetching TMH data ...'
    train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(dataset='tmh')


def train_ffnn(dropout=False):
    """Create, compile and train feedforward neural net."""
    x = T.fmatrix()
    y = T.imatrix()
    fc1 = FullyConnectedLayer(
        input_dim=train_x.shape[1],
        output_dim=500,
        activation='relu'
    )
    fc2 = FullyConnectedLayer(
        input_dim=500,
        output_dim=500,
        activation='relu'
    )
    fc3 = FullyConnectedLayer(
        input_dim=500,
        output_dim=10,
        activation='softmax'
    )
    if dropout:
        drp1 = DropoutLayer()
        drp2 = DropoutLayer()
    params = fc1.params + fc2.params + fc3.params
    act1 = fc1.fprop(x)
    if dropout:
        act1 = drp1.fprop(act1)
    act2 = fc2.fprop(act1)
    if dropout:
        act2 = drp2.fprop(act2)
    act3 = fc3.fprop(act2)
    loss = T.nnet.categorical_crossentropy(
        act3,
        y
    ).mean()

    print 'Compiling optimization method ...'
    updates = Optimizer().sgd(
        loss,
        params,
        lr=0.01
    )

    print 'Compiling train function ...'
    f_train = theano.function(
        inputs=[x, y],
        outputs=loss,
        updates=updates
    )

    print 'Compiling evaluation function ...'
    f_eval = theano.function(
        inputs=[x],
        outputs=act3
    )

    print 'Training network ...'
    for epoch in xrange(10):
        costs = []
        for batch in xrange(0, train_x.shape[0], 24):
            cost = f_train(
                train_x[batch:batch + 24],
                train_y[batch:batch + 24]
            )
            costs.append(cost)
        print 'Epoch %d Training Loss : %.3f ' % (epoch, np.mean(costs))

    dev_predictions = f_eval(dev_x)
    test_predictions = f_eval(test_x)
    print 'Accuracy on dev : %.3f%% ' % (
        100. * (np.argmax(dev_predictions, axis=1) == dev_y).mean()
    )
    print 'Accuracy on test : %.3f%% ' % (
        100. * (np.argmax(test_predictions, axis=1) == test_y).mean()
    )


def train_cnn(
    train_x,
    train_y,
    dev_x,
    dev_y,
    test_x,
    test_y
):
    """Train CNN."""
    train_x = train_x.reshape(
        train_x.shape[0],
        1,
        int(np.sqrt(train_x.shape[1])),
        int(np.sqrt(train_x.shape[1]))
    )

    dev_x = dev_x.reshape(
        dev_x.shape[0],
        1,
        int(np.sqrt(dev_x.shape[1])),
        int(np.sqrt(dev_x.shape[1]))
    )
    test_x = test_x.reshape(
        test_x.shape[0],
        1,
        int(np.sqrt(test_x.shape[1])),
        int(np.sqrt(test_x.shape[1]))
    )

    x = T.tensor4()
    y = T.imatrix()

    convolution_layer0 = Convolution2DLayer(
        num_kernels=20,
        num_channels=1,
        kernel_width=5,
        kernel_height=5,
        batch_normalization=True
    )

    convolution_layer1 = Convolution2DLayer(
        num_kernels=50,
        num_channels=20,
        kernel_width=5,
        kernel_height=5,
        batch_normalization=True
    )

    kmax_layer = KMaxPoolingLayer(
        pooling_factor=(2, 2)
    )

    fc1 = FullyConnectedLayer(5000, 500, activation='relu')
    fc2 = FullyConnectedLayer(500, 10, activation='softmax')

    params = convolution_layer0.params + convolution_layer1.params + \
        fc1.params + fc2.params
    act1 = convolution_layer0.fprop(x)
    print 'conv 1', act1.eval({x: np.random.rand(2, 1, 24, 24).astype(np.float32)}).shape
    act2 = convolution_layer1.fprop(act1)
    print 'conv 2', act2.eval({x: np.random.rand(2, 1, 24, 24).astype(np.float32)}).shape
    act3 = kmax_layer.fprop(act2)
    act3 = act3.flatten(ndim=2)
    act4 = fc1.fprop(act3)
    act5 = fc2.fprop(act4)
    loss = T.nnet.categorical_crossentropy(
        act5,
        y
    ).mean()

    print 'Compiling optimization method ...'
    updates = Optimizer().sgd(
        loss,
        params,
        lr=0.01
    )

    print 'Compiling train function ...'
    f_train = theano.function(
        inputs=[x, y],
        outputs=loss,
        updates=updates
    )

    print 'Compiling evaluation function ...'
    f_eval = theano.function(
        inputs=[x],
        outputs=act5
    )

    print 'Training network ...'
    for epoch in xrange(10):
        costs = []
        for batch in xrange(0, train_x.shape[0], 24):
            cost = f_train(
                train_x[batch:batch + 24],
                train_y[batch:batch + 24]
            )
            costs.append(cost)
            print 'Finished minibatch %d cost : %.3f ' % (batch, cost)
        print 'Epoch %d Training Loss : %.3f ' % (epoch, np.mean(costs))

    dev_predictions = f_eval(dev_x)
    test_predictions = f_eval(test_x)
    print 'Accuracy on dev : %.3f%% ' % (
        100. * (np.argmax(dev_predictions, axis=1) == dev_y).mean()
    )
    print 'Accuracy on test : %.3f%% ' % (
        100. * (np.argmax(test_predictions, axis=1) == test_y).mean()
    )


def train_rnn(network):
    """Train CNN."""
    x = T.ivector()
    y = T.ivector()

    emb = EmbeddingLayer(20, 50, name='embedding')
    if network == 'rnn':
        rnn = RNN(50, 50, name='rnn')
    elif network == 'lstm':
        rnn = LSTM(50, 50, name='lstm')
    fc1 = FullyConnectedLayer(50, 1, name='fc')

    params = emb.params + rnn.params + \
        fc1.params
    embs = emb.fprop(x)
    act1 = rnn.fprop(embs)
    act2 = fc1.fprop(act1)
    loss = ((act2.transpose() - y) ** 2).mean()

    print 'Compiling optimization method ...'
    updates = Optimizer().sgd(
        loss,
        params,
        lr=0.01
    )

    print 'Compiling train function ...'
    f_train = theano.function(
        inputs=[x, y],
        outputs=loss,
        updates=updates
    )

    print 'Compiling evaluation function ...'
    f_eval = theano.function(
        inputs=[x],
        outputs=act2
    )

    print 'Training network ...'
    for epoch in xrange(10):
        costs = []
        for data_point, labels in zip(train_x, train_y):
            cost = f_train(
                data_point,
                labels
            )
        costs.append(cost)

        print 'Epoch %d Training Loss : %f ' % (epoch, np.mean(costs))

    accs = []
    for data_point, labels in zip(test_x, test_y):
        preds = f_eval(data_point).squeeze()
        preds = [1 if pred > 0.5 else 0 for pred in preds]
        acc = sum([True if a == b else False for a, b in zip(preds, labels)]) \
            / float(len(preds))
        accs.append(acc)
    print 'Testing Accuracy : %f%% ' % (np.mean(accs) * 100.)

if layer == 'cnn':
    train_cnn(
        train_x,
        train_y,
        dev_x,
        dev_y,
        test_x,
        test_y
    )
elif layer == 'fc':
    train_ffnn()
elif layer == 'rnn':
    train_rnn(network='rnn')
elif layer == 'lstm':
    train_rnn(network='lstm')
elif layer == 'drp':
    train_ffnn(dropout=True)
