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


class BatchNormalizationLayer:
    """Batch Normalization core."""

    # Inspired by https://github.com/shuuki4/Batch-Normalization

    def __init__(self, input_shape, layer='fc', momentum=0.9):
        """Initialize parameters."""
        self.input_shape = input_shape
        self.layer = layer  # Can be fc/conv
        self.momentum = momentum
        self.run_mode = 0  # 0/1 training/testing
        self.epsilon = 1e-6

        self.input_size = input_shape[1]
        # random setting of gamma and beta, setting initial mean and std
        self.gamma = theano.shared(
            np.asarray(np.random.uniform(
                low=-1.0 / np.sqrt(self.input_size),
                high=1.0 / np.sqrt(self.input_size),
                size=(input_shape[1])),
            ).astype(np.float32),
            name='gamma',
            borrow=True
        )
        self.beta = theano.shared(
            np.zeros(
                (self.input_size),
            ).astype(np.float32),
            name='beta',
            borrow=True
        )
        self.mean = theano.shared(
            np.zeros(
                (self.input_size),
                dtype=theano.config.floatX
            ),
            name='mean',
            borrow=True
        )

        self.var = theano.shared(
            np.ones(
                (input_shape[1]),
                dtype=theano.config.floatX
            ),
            name='var',
            borrow=True
        )

        # parameter save for update
        self.params = [self.gamma, self.beta]

    def set_runmode(self, run_mode):
        """Change the running mode."""
        self.run_mode = run_mode

    def change_shape(self, vec):
        """Modify shape of batch norm params."""
        vec = T.repeat(vec, self.input_shape[2] * self.input_shape[3])
        return vec.reshape(
            (self.input_shape[1], self.input_shape[2], self.input_shape[3])
        )

    def fprop(self, input):
        """"Propogate input through the layer."""
        if self.layer == 'fc':
            # Training time
            if self.run_mode == 0:
                mean_t = T.mean(input, axis=0)  # Compute mean
                var_t = T.var(input, axis=0)  # Compute variance
                # Subtract mean and divide by std
                norm_t = (input - mean_t) / T.sqrt(var_t + self.epsilon)
                # Add parameters
                output = self.gamma * norm_t + self.beta
                # Update mean and variance
                self.mean = self.momentum * self.mean + \
                    (1.0 - self.momentum) * mean_t
                self.var = self.momentum * self.var + (1.0 - self.momentum) \
                    * (self.input_shape[0] / (self.input_shape[0] - 1) * var_t)
            # Test time - use statistics from the training data
            else:
                output = self.gamma * (input - self.mean) / \
                    T.sqrt(self.var + self.epsilon) + self.beta

        elif self.layer == 'conv':
            if self.run_mode == 0:
                # Mean across every channel
                mean_t = T.mean(input, axis=(0, 2, 3))
                var_t = T.var(input, axis=(0, 2, 3))
                # mean, var update
                self.mean = self.momentum * self.mean + \
                    (1.0 - self.momentum) * mean_t
                self.var = self.momentum * self.var + (1.0-self.momentum) * \
                    (self.input_shape[0] / (self.input_shape[0] - 1) * var_t)
            else:
                mean_t = self.mean
                var_t = self.var
            # change shape to fit input shape
            mean_t = self.change_shape(mean_t)
            var_t = self.change_shape(var_t)
            gamma_t = self.change_shape(self.gamma)
            beta_t = self.change_shape(self.beta)

            output = gamma_t * (input - mean_t) / \
                T.sqrt(var_t + self.epsilon) + beta_t
        return output


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
