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
        batch_normalization=True,
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
            self.activation = None
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
                std=linear_activation.var(keepdims=True),
                mode='low_mem',
            ).astype(theano.config.floatX)
        if self.activation == 'linear':
            return linear_activation
        else:
            return self.activation(linear_activation)


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


class SoftMaxLayer:
    """Softmax Layer."""

    def fprop(self, input):
        """Propogate input through the layer."""
        return T.nnet.softmax(input)


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


class MaxoutLayer:
    """Maxout Layer."""

    # http://jmlr.csail.mit.edu/proceedings/papers/v28/goodfellow13.pdf
    # THIS IS STILL EXPERIMENTAL AND DOES NOT WORK
    def __init__(
        self,
        input_dim,
        pool_size,
        dropout_rate=None,
        name='maxout'
    ):
        """Initialize dropout rate, and weights."""
        self.input_dim = input_dim
        self.pool_size = pool_size
        self.output_dim = np.ceil(self.input_dim / self.pool_size)
        self.is_training = True
        self.dropout_rate = dropout_rate
        self.rng = np.random.RandomState(1234)
        self.srng = T.shared_randomstreams.RandomStreams(
            self.rng.randint(1337)
        )

        self.weights = get_weights(
            shape=(self.input_dim, self.input_dim),
            name=name + '__weights'
        )
        self.bias = get_bias(self.input_dim, name=name + '__bias')

        self.params = [self.weights, self.bias]

    def fprop(self, input, is_training=True):
        """Propogate the input through the layer."""
        cur_max = None
        linear_activation = T.dot(input, self.weights) + self.bias

        # If dropout is enabled and the network is training,
        # use the dropout mask before applying the non-linearity
        if self.dropout_rate is not None and self.is_training:
            dropout_mask = self.srng.binomial(
                n=1,
                p=1.0-self.dropout_rate,
                size=linear_activation.shape,
                dtype=theano.config.floatX
            )
            linear_activation = linear_activation * dropout_mask

        # If dropout is enabled and the network is being used for predictions,
        # don't apply the dropout mask and scale the activation by the dropout
        elif self.dropout_rate is not None and not is_training:
            linear_activation = linear_activation / self.dropout_rate

        for i in xrange(self.pool_size):
            activation_subset = linear_activation[:, i::self.pool_size]
            if cur_max is None:
                cur_max = activation_subset
            else:
                cur_max = T.maximum(cur_max, activation_subset)

        return cur_max


class DropConnectLayer:
    """DropConnect Layer."""

    # http://www.matthewzeiler.com/pubs/icml2013/icml2013.pdf
    # THIS IS EXPERIMENT AND HASN'T BEEN COMPLETED YET

    def __init__(
        self,
        input_dim,
        output_dim,
        drop_rate=0.3,
        activation='sigmoid',
        name='dropconnect'
    ):
        """Initialize weights, biases and dropconnect rate."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_rate = drop_rate

        # Set the activation function for this layer
        if activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'linear':
            self.activation = None
        else:
            raise NotImplementedError("Unknown activation")

        # Initialize weights & biases for this layer
        self.weights = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__weights'
        )
        self.bias = get_bias(output_dim, name=name + '__bias')
        self.params = [self.weights, self.bias]


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
