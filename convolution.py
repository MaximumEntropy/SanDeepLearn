"""Implementations of convolution layers."""
from utils import get_weights, get_bias

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from layer import FullyConnectedLayer, BatchNormalizationLayer
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
        name='conv',
        b=True
    ):
        """Intialize convolution filters."""
        self.num_kernels = num_kernels
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.stride = stride
        self.border_mode = border_mode
        self.num_channels = num_channels
        self.b = b
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
        self.kernel_shape = (
            self.num_kernels,
            self.num_channels,
            self.kernel_height,
            self.kernel_width
        )

        # Get kernels and bias
        strategy = 'he2015' if activation == 'relu' else 'glorot'
        self.kernels = get_weights(
            shape=self.kernel_shape,
            name=name + '__kernels',
            strategy=strategy
        )
        self.bias = get_bias(self.num_kernels, name=name + '__bias')
        if self.b:
            self.params = [self.kernels, self.bias]
        else:
            self.params = [self.kernels]

    def fprop(self, input):
        """Propogate the input through the layer."""
        self.convolution = T.nnet.conv2d(
            input=input,
            filters=self.kernels,
            filter_shape=self.kernel_shape,
            subsample=self.stride,
            border_mode=self.border_mode
        )

        self.conv_out = self.convolution if not self.b else \
            self.convolution + self.bias.dimshuffle('x', 0, 'x', 'x')

        return self.conv_out if self.activation == 'linear' else \
            self.activation(self.conv_out)


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

    def __init__(
        self,
        num_kernels,
        num_channels,
        kernel_height,
        kernel_width,
        input_shapes,
        name,
        increase_dim=False,
        first=False
    ):
        """Initialize residual block params."""
        self.num_kernels = num_kernels
        self.num_channels = num_channels
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.input_shapes = input_shapes
        self.name = name
        self.increase_dim = increase_dim
        self.first_stride = (2, 2) if self.increase_dim else (1, 1)
        self.first = first

        self.conv1 = Convolution2DLayer(
            num_kernels=self.num_kernels[0],
            num_channels=self.num_channels[0],
            kernel_height=self.kernel_height[0],
            kernel_width=self.kernel_width[0],
            stride=self.first_stride,
            border_mode='half',
            activation='relu',
            name='conv_block_%d_layer_%d' % (self.name, 0)
        )

        self.conv2 = Convolution2DLayer(
            num_kernels=self.num_kernels[1],
            num_channels=self.num_channels[1],
            kernel_height=self.kernel_height[1],
            kernel_width=self.kernel_width[1],
            stride=(1, 1),
            border_mode='half',
            activation='linear',
            name='conv_block_%d_layer_%d' % (self.name, 1)
        )

        self.bn1 = BatchNormalizationLayer(
            input_shape=self.input_shapes[0],
            layer='conv'
        )

        self.bn2 = BatchNormalizationLayer(
            input_shape=self.input_shapes[1],
            layer='conv'
        )

        self.proj_kernels = get_weights(
            shape=(num_kernels[0], num_channels[0], 1, 1),
            name='__proj_kernels'
        )

        self.conv_layers = [self.conv1, self.conv2]
        self.bn_layers = [self.bn1, self.bn2]
        self.params = []

        if self.first:
            self.params = self.conv1.params + self.conv2.params + \
                self.bn2.params
        elif self.increase_dim:
            self.params = self.conv1.params + self.conv2.params + self.bn1.params \
                + self.bn2.params + [self.proj_kernels]
        else:
            self.params = self.conv1.params + self.conv2.params + self.bn1.params \
                + self.bn2.params

    def fprop(self, input):
        """Propogate input through the network."""
        if self.first:
            bn_pre_relu = input
        else:
            bn_pre_conv = self.bn1.fprop(input)
            bn_pre_relu = T.nnet.relu(bn_pre_conv)

        self.conv_1 = self.conv1.fprop(bn_pre_relu)
        self.conv_1 = self.bn2.fprop(self.conv_1)

        self.conv_2 = self.conv2.fprop(self.conv_1)

        if self.increase_dim:
            # projection shortcut, as option B in paper
            self.projection = T.nnet.conv2d(
                input=input,
                filters=self.proj_kernels,
                border_mode='half',
                subsample=(2, 2),
            )
            output = T.add(self.conv_2, self.projection)
        else:
            output = T.add(self.conv_2, input)

        return output


class VGGNetwork:
    """VGG Convnet."""

    def __init__(self, path_to_weights, pretrained=True):
        """Initialize convolution layers."""
        self.path_to_weights = path_to_weights
        self.pretrained = pretrained
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

        if self.pretrained:
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
        self.output_pool5 = self.pool5.fprop(self.output_conv5_4)

        self.output_fc6 = self.full1.fprop(
            self.output_pool5.reshape(
                (input.shape[0], 512 * 7 * 7)
            )
        )
        self.output_fc7 = self.full2.fprop(self.output_fc6)
        self.output_fc8 = self.full3.fprop(self.output_fc7)
