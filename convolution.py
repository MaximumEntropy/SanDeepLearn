"""Implementations of convolution layers."""
from utils import get_weights, get_bias

import theano
import theano.tensor as T
from theano.tensor.signal import pool
import numpy as np
from layer import FullyConnectedLayer
import pickle
theano.config.floatX = 'float32'

__author__ = "Sandeep Subramanian"
__maintainer__ = "Sandeep Subramanian"
__version__ = "1.0"
__email__ = "sandeep.subramanian@gmail.com"


class Convolution2DLayer:
    """2D Convolution Layer."""

    def __init__(
        self,
        kernel_width,
        kernel_height,
        num_kernels,
        num_channels,
        activation='relu',
        stride=(1, 1),
        border_mode='valid',
        batch_normalization=False,
        name='conv',
    ):
        """Intialize convolution filters."""
        self.num_kernels = num_kernels
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.stride = stride
        self.border_mode = border_mode
        self.batch_normalization = batch_normalization
        self.num_channels = num_channels
        # Compute fan_in and fan_out to initialize filters
        self.fan_in = num_channels * kernel_width * kernel_height
        self.fan_out = num_kernels * kernel_width * kernel_height

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

        # Compute the output shape of the network
        '''
        if self.wide:
            self.output_height_shape = (
                self.input_height +
                self.kernel_height - 1
            ) / self.stride[0]
            self.output_width_shape = (
                self.input_width +
                self.kernel_width - 1
            ) / self.stride[1]
        elif not self.wide:
            self.output_height_shape = (
                self.input_height -
                self.kernel_height + 1
            ) / self.stride[0]
            self.output_width_shape = (
                self.input_width -
                self.kernel_width + 1
            ) / self.stride[1]
        '''
        self.kernel_shape = (
            self.num_kernels,
            self.num_channels,
            self.kernel_height,
            self.kernel_width
        )

        # Get kernels and bias
        self.kernels = get_weights(
            shape=self.kernel_shape,
            name=name + '__kernels'
        )
        self.bias = get_bias(self.num_kernels, name=name + '__bias')
        if self.batch_normalization:
            self.gamma = theano.shared(
                value=np.ones((
                    self.num_kernels,
                )),
                name='gamma'
            )
            self.beta = theano.shared(
                value=np.ones((
                    self.num_kernels,
                )),
                name='beta'
            )
            self.params = [self.kernels, self.bias, self.gamma, self.beta]
        else:
            self.params = [self.kernels, self.bias]

    def fprop(self, input):
        """Propogate the input through the layer."""
        self.convolution = T.nnet.conv2d(
            input=input,
            filters=self.kernels,
            filter_shape=self.kernel_shape,
            subsample=self.stride,
            border_mode=self.border_mode
        )

        self.conv_out = self.convolution + \
            self.bias.dimshuffle('x', 0, 'x', 'x')
        if self.batch_normalization:
            self.conv_out = T.nnet.bn.batch_normalization(
                inputs=self.conv_out,
                gamma=self.gamma,
                beta=self.beta,
                mean=self.conv_out.mean(axis=1).dimshuffle(0, 'x', 1, 2),
                std=self.conv_out.std(axis=1).dimshuffle(0, 'x', 1, 2),
                mode='low_mem',
            ).astype(theano.config.floatX)

        return self.conv_out if self.activation == 'linear' \
            else self.activation(self.conv_out)


class KMaxPoolingLayer:
    """K-Max Pooling Layer."""

    def __init__(
        self,
        pooling_factor=(2, 2),
        stride=(2, 2),
        name='kmaxpooling'
    ):
        """Set pooling factor."""
        self.pooling_factor = pooling_factor
        self.stride = stride

    def fprop(self, input):
        """Propogate the input through the layer."""
        return pool.pool_2d(
            input=input,
            ds=self.pooling_factor,
            st=self.stride,
            ignore_border=True
        )


class ConvResidualBlock:
    """Residual block of convnets."""

    def __init__(self, conv_layers):
        """Initialize residual block params."""
        self.conv_layers = conv_layers
        self.num_layers = len(conv_layers)
        self.params = []
        for conv_layer in self.conv_layers:
            self.params += conv_layer.params

    def fprop(self, input):
        """Propogate input through the network."""
        prev_inp = input
        for conv_layer in self.conv_layers[:-1]:
            prev_inp = conv_layer.fprop(prev_inp)

        projection = self.conv_layers[-1].fprop(input)
        return T.nnet.relu(prev_inp + projection)


class VGGNetwork:
    """VGG Convnet."""

    def __init__(self, path_to_weights):
        """Initialize convolution layers."""
        self.path_to_weights = path_to_weights
        self.conv1_1 = Convolution2DLayer(
            num_kernels=64,
            num_channels=3,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv1_1'
        )
        self.conv1_2 = Convolution2DLayer(
            num_kernels=64,
            num_channels=64,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv1_2'
        )
        self.pool1 = KMaxPoolingLayer(
            pooling_factor=(2, 2)
        )

        self.conv2_1 = Convolution2DLayer(
            num_kernels=128,
            num_channels=64,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv2_1'
        )
        self.conv2_2 = Convolution2DLayer(
            num_kernels=128,
            num_channels=128,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv2_2'
        )
        self.pool2 = KMaxPoolingLayer(
            pooling_factor=(2, 2)
        )

        self.conv3_1 = Convolution2DLayer(
            num_kernels=256,
            num_channels=128,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv3_1'
        )
        self.conv3_2 = Convolution2DLayer(
            num_kernels=256,
            num_channels=256,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv3_2'
        )
        self.conv3_3 = Convolution2DLayer(
            num_kernels=256,
            num_channels=256,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv3_3'
        )
        self.conv3_4 = Convolution2DLayer(
            num_kernels=256,
            num_channels=256,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv3_4'
        )
        self.pool3 = KMaxPoolingLayer(
            pooling_factor=(2, 2)
        )

        self.conv4_1 = Convolution2DLayer(
            num_kernels=512,
            num_channels=256,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv4_1'
        )
        self.conv4_2 = Convolution2DLayer(
            num_kernels=512,
            num_channels=512,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv4_2'
        )
        self.conv4_3 = Convolution2DLayer(
            num_kernels=512,
            num_channels=512,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv4_3'
        )
        self.conv4_4 = Convolution2DLayer(
            num_kernels=512,
            num_channels=512,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv4_4'
        )
        self.pool4 = KMaxPoolingLayer(
            pooling_factor=(2, 2)
        )

        self.conv5_1 = Convolution2DLayer(
            num_kernels=512,
            num_channels=512,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv5_1'
        )
        self.conv5_2 = Convolution2DLayer(
            num_kernels=512,
            num_channels=512,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv5_2'
        )
        self.conv5_3 = Convolution2DLayer(
            num_kernels=512,
            num_channels=512,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv5_3'
        )
        self.conv5_4 = Convolution2DLayer(
            num_kernels=512,
            num_channels=512,
            kernel_height=3,
            kernel_width=3,
            border_mode='half',
            name='conv5_4'
        )
        self.pool5 = KMaxPoolingLayer(
            pooling_factor=(2, 2)
        )

        self.full1 = FullyConnectedLayer(
            input_dim=512 * 7 * 7,
            output_dim=4096,
            activation='relu',
            batch_normalization=False,
            name='fc6'
        )
        self.full2 = FullyConnectedLayer(
            input_dim=4096,
            output_dim=4096,
            activation='relu',
            batch_normalization=False,
            name='fc7'
        )
        self.full3 = FullyConnectedLayer(
            input_dim=4096,
            output_dim=1000,
            activation='softmax',
            batch_normalization=False,
            name='fc8'
        )

        self.params = []
        self.params += self.conv1_1.params + self.conv1_2.params
        self.params += self.conv2_1.params + self.conv2_2.params
        self.params += self.conv3_1.params + self.conv3_2.params + \
            self.conv3_3.params + self.conv3_4.params
        self.params += self.conv4_1.params + self.conv4_2.params + \
            self.conv4_3.params + self.conv4_4.params
        self.params += self.conv5_1.params + self.conv5_2.params + \
            self.conv5_3.params + self.conv5_4.params
        self.params += self.full1.params + self.full2.params \
            + self.full3.params

        print 'Initializing network weights ...'
        self._set_weights()

    def _set_weights(self):
        pretrained_model = pickle.load(open(self.path_to_weights))
        assert len(pretrained_model['param values']) == len(self.params)
        for pretrained_values, param in zip(
            pretrained_model['param values'],
            self.params
        ):
            assert pretrained_values.shape == param.get_value().shape
            param.set_value(pretrained_values)

    def fprop(self, input):
        """Propogate the input through the network."""
        self.output_conv1_1 = self.conv1_1.fprop(input)
        self.output_conv1_2 = self.conv1_2.fprop(self.output_conv1_1)
        self.output_pool1 = self.pool1.fprop(self.output_conv1_2)

        self.output_conv2_1 = self.conv2_1.fprop(self.output_pool1)
        self.output_conv2_2 = self.conv2_2.fprop(self.output_conv2_1)
        self.output_pool2 = self.pool2.fprop(self.output_conv2_2)

        self.output_conv3_1 = self.conv3_1.fprop(self.output_pool2)
        self.output_conv3_2 = self.conv3_2.fprop(self.output_conv3_1)
        self.output_conv3_3 = self.conv3_3.fprop(self.output_conv3_2)
        self.output_conv3_4 = self.conv3_4.fprop(self.output_conv3_3)
        self.output_pool3 = self.pool3.fprop(self.output_conv3_4)

        self.output_conv4_1 = self.conv4_1.fprop(self.output_pool3)
        self.output_conv4_2 = self.conv4_2.fprop(self.output_conv4_1)
        self.output_conv4_3 = self.conv4_3.fprop(self.output_conv4_2)
        self.output_conv4_4 = self.conv4_4.fprop(self.output_conv4_3)
        self.output_pool4 = self.pool4.fprop(self.output_conv4_4)

        self.output_conv5_1 = self.conv5_1.fprop(self.output_pool4)
        self.output_conv5_2 = self.conv5_2.fprop(self.output_conv5_1)
        self.output_conv5_3 = self.conv5_3.fprop(self.output_conv5_2)
        self.output_conv5_4 = self.conv5_4.fprop(self.output_conv5_3)
        self.output_pool5 = self.pool5.fprop(self.output_conv5_3)

        self.output_fc6 = self.full1.fprop(
            self.output_pool5.reshape(
                (input.shape[0], 512 * 7 * 7)
            )
        )
        self.output_fc7 = self.full2.fprop(self.output_fc6)
        self.output_fc8 = self.full3.fprop(self.output_fc7)
