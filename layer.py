"""Implementations of feedforward layers."""
from utils import get_weights, get_bias, get_highway_bias, get_relu_weights

import theano
import theano.tensor as T
import numpy as np

__author__ = "Sandeep Subramanian"
__maintainer__ = "Sandeep Subramanian"
__version__ = "1.0"
__email__ = "sandeep.subramanian@gmail.com"


class FullyConnectedLayer:
    """Fully Connected Layer."""

    def __init__(
        self,
        input_dim,
        output_dim,
        activation='sigmoid',
        batch_normalization=False,
        name='fully_connected'
    ):
        """Initialize weights and bias."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_normalization = batch_normalization

        # Set the activation function for this layer
        if activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'relu':
            self.activation = T.nnet.relu
        elif activation == 'softmax':
            self.activation = T.nnet.softmax
        elif activation == 'linear':
            self.activation = 'linear'
        else:
            raise NotImplementedError("Unknown activation")

        # Initialize weights & biases for this layer
        if activation == 'relu':
            self.weights = get_relu_weights(
                (input_dim, output_dim),
                name=name + '__weights'
            )
        else:
            self.weights = get_weights(
                shape=(input_dim, output_dim),
                name=name + '__weights'
            )

        self.bias = get_bias(output_dim, name=name + '__bias')
        if self.batch_normalization:
            self.gamma = theano.shared(
                value=np.ones((self.output_dim,)),
                name='gamma'
            )
            self.beta = theano.shared(
                value=np.zeros((self.output_dim,)),
                name='beta'
            )
            self.params = [self.weights, self.bias, self.gamma, self.beta]
        else:
            self.params = [self.weights, self.bias]

    def fprop(self, input):
        """Propogate the input through the FC-layer."""
        linear_activation = T.dot(input, self.weights) + self.bias
        if self.batch_normalization:
            linear_activation = T.nnet.bn.batch_normalization(
                inputs=linear_activation,
                gamma=self.gamma,
                beta=self.beta,
                mean=linear_activation.mean(keepdims=True),
                std=linear_activation.std(keepdims=True),
                mode='low_mem',
            ).astype(theano.config.floatX)
        if self.activation == 'linear':
            return linear_activation
        else:
            return self.activation(linear_activation)

'''
class DropoutLayer:
    """Dropout Layer."""

    # https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

    def __init__(self, dropout_rate=0.5):
        """Set dropout rate."""
        self.dropout_rate = dropout_rate
        self.rng = np.random.RandomState(1234)
        self.srng = T.shared_randomstreams.RandomStreams(
            self.rng.randint(1337)
        )

    def fprop(self, input, deterministic=False):
        """Apply dropout mask to the input."""
        dropout_mask = self.srng.binomial(
            n=1,
            p=self.dropout_rate,
            size=input.shape,
            dtype=theano.config.floatX
        )
        output = T.switch()
        return input * dropout_mask
'''


class SoftMaxLayer:
    """Softmax Layer."""

    def fprop(self, input):
        """Propogate input through the layer."""
        return T.nnet.softmax(input)


class SoftMaxLayer3D:
    """3D Softmax Layer."""

    def fprop(self, input):
        """Propogate input through the layer."""
        e = T.exp(input - T.max(input, axis=-1, keepdims=True))
        s = T.sum(e, axis=-1, keepdims=True)
        return e / s


class EmbeddingLayer:
    """Embedding layer lookup table."""

    def __init__(
        self,
        input_dim,
        output_dim,
        pretrained=None,
        name='embedding'
    ):
        """Initialize random embedding matrix."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pretrained = pretrained

        if self.pretrained is not None:
            assert input_dim == pretrained.shape[0] and \
                output_dim == pretrained.shape[1]
            self.embedding = theano.shared(
                pretrained.astype(np.float32),
                borrow=True,
                name=name + '__pretrained_embedding'
            )
        else:
            self.embedding = theano.shared(
                np.random.uniform(
                    low=-1.0,
                    high=1.0,
                    size=(input_dim, output_dim)
                ).astype(np.float32),
                name=name + '__embedding',
                borrow=True
            )
        self.params = [self.embedding]

    def fprop(self, input):
        """Propogate the input through the layer."""
        return self.embedding[input]


class FullyConnectedResidualBlock:
    """Residual block of FC layers."""

    def __init__(self, fc_layers):
        """Initialize residual block params."""
        self.fc_layers = fc_layers
        self.num_layers = len(fc_layers)
        self.params = []
        for layer in self.fc_layers:
            self.params += layer.params

    def fprop(self, input):
        """Propogate input through the network."""
        prev_inp = input
        for layer in self.fc_layers:
            prev_inp = layer.fprop(prev_inp)
        return T.nnet.relu(prev_inp + input)


class HighwayNetworkLayer:
    """HighwayNetwork Layer."""

    # http://arxiv.org/pdf/1505.00387v2.pdf
    # http://arxiv.org/pdf/1507.06228v2.pdf

    # THIS IS EXPERIMENTAL AND DOES NOT WORK RIGHT NOW

    def __init__(
        self,
        input_dim,
        output_dim,
        activation='sigmoid',
        name='fully_connected'
    ):
        """Initialize weights and biases."""
        # Set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Set the activation function for this layer
        if activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'relu':
            self.activation = T.nnet.relu
        elif activation == 'linear':
            self.activation = None
        else:
            raise NotImplementedError("Unknown activation")

        self.weights = get_relu_weights(
            (input_dim, output_dim),
            name=name + '__weights'
        )
        self.gating_weights = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__weights'
        )  # Transform gate

        self.bias = get_bias(output_dim, name=name + '__bias')
        self.gating_bias = get_highway_bias(
            output_dim,
            name=name + '__gating_bias'
        )
        self.params = [
            self.weights,
            self.gating_weights,
            self.gating_bias,
            self.bias
        ]

    def fprop(self, input):
        """Propogate the input through the layer."""
        gate_activation = T.nnet.sigmoid(
            T.dot(input, self.gating_weights) +
            self.gating_bias
        )
        layer_activation = self.activation(
            T.dot(input, self.weights) +
            self.bias
        )

        return layer_activation * gate_activation * (1.0 - gate_activation)
