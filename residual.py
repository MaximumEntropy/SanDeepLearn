"""Implementations of residual architectures in Lasagne."""

# https://github.com/FlorianMuellerklein/Identity-Mapping-ResNet-Lasagne
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import InputLayer, DenseLayer, \
    batch_norm, BatchNormLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer, NonlinearityLayer, \
    GlobalPoolLayer
from lasagne.init import HeNormal


class ConvResnet110:
    """Preactivation residual block."""

    # https://arxiv.org/abs/1603.05027
    def __init__(
        self,
        input_symbol,
        n=18,
        num_channels=3,
        img_dim=32
    ):
        """Initialize block parameters."""
        self.input_symbol = input_symbol
        self.n = n
        self.num_channels = num_channels
        self.img_dim = img_dim
        self.residual_blocks = []

    def preactivation_residual_block(
        self,
        input,
        increase_dim=False,
        projection=False,
        first=False
    ):
        """Residual Preactivation conv block."""
        input_num_filters = input.output_shape[1]
        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        if first:
            bn_pre_relu = input
        else:
            bn_pre_conv = BatchNormLayer(input)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        conv_1 = batch_norm(ConvLayer(
            bn_pre_relu,
            num_filters=out_num_filters,
            filter_size=(3, 3),
            stride=first_stride,
            nonlinearity=rectify,
            pad='same',
            W=HeNormal(gain='relu')
        ))

        conv_2 = ConvLayer(
            conv_1,
            num_filters=out_num_filters,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=None,
            pad='same',
            W=HeNormal(gain='relu')
        )

        if increase_dim:
            projection = ConvLayer(
                bn_pre_relu,
                num_filters=out_num_filters,
                filter_size=(1, 1),
                stride=(2, 2),
                nonlinearity=None,
                pad='same',
                b=None
            )
            block = ElemwiseSumLayer([conv_2, projection])
        else:
            block = ElemwiseSumLayer([conv_2, input])

        return block

    def bottleneck_residual_block(
        self,
        input,
        increase_dim=False,
        first=False
    ):
        """Bottleneck implementation of residual block."""
        input_num_filters = input.output_shape[1]

        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        if first:
            bn_pre_relu = input
            out_num_filters = out_num_filters * 4
        else:
            bn_pre_conv = BatchNormLayer(input)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        bottleneck_filters = out_num_filters / 4

        conv_1 = batch_norm(ConvLayer(
            bn_pre_relu,
            num_filters=bottleneck_filters,
            filter_size=(1, 1),
            stride=(1, 1),
            nonlinearity=rectify,
            pad='same',
            W=HeNormal(gain='relu')
        ))

        conv_2 = batch_norm(ConvLayer(
            conv_1,
            num_filters=bottleneck_filters,
            filter_size=(3, 3),
            stride=first_stride,
            nonlinearity=rectify,
            pad='same',
            W=HeNormal(gain='relu')
        ))

        conv_3 = ConvLayer(
            conv_2,
            num_filters=out_num_filters,
            filter_size=(1, 1),
            stride=(1, 1),
            nonlinearity=None,
            pad='same',
            W=HeNormal(gain='relu')
        )

        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(
                bn_pre_relu,
                num_filters=out_num_filters,
                filter_size=(1, 1),
                stride=(2, 2),
                nonlinearity=None,
                pad='same',
                b=None
            )
            block = ElemwiseSumLayer([conv_3, projection])

        elif first:
            # projection shortcut, as option B in paper
            projection = ConvLayer(
                input,
                num_filters=out_num_filters,
                filter_size=(1, 1),
                stride=(1, 1),
                nonlinearity=None,
                pad='same',
                b=None
            )
            block = ElemwiseSumLayer([conv_3, projection])

        else:
            block = ElemwiseSumLayer([conv_3, input])

        return block

    def construct_preactivation_network(self):
        """Construct the entire resnet with preactivation blocks."""
        input = InputLayer(
            shape=(None, 3, self.img_dim, self.img_dim),
            input_var=self.input_symbol
        )

        output = batch_norm(ConvLayer(
            input,
            num_filters=16,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=rectify,
            pad='same',
            W=HeNormal(gain='relu')
        ))

        # first stack of residual blocks, output is 16 x 32 x 32
        output = self.preactivation_residual_block(output, first=True)
        for _ in range(1, self.n):
            output = self.preactivation_residual_block(output)

        # second stack of residual blocks, output is 32 x 16 x 16
        output = self.preactivation_residual_block(output, increase_dim=True)
        for _ in range(1, self.n):
            output = self.preactivation_residual_block(output)

        # third stack of residual blocks, output is 64 x 8 x 8
        output = self.preactivation_residual_block(output, increase_dim=True)
        for _ in range(1, self.n):
            output = self.preactivation_residual_block(output)

        bn_post_conv = BatchNormLayer(output)
        bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

        # average pooling
        avg_pool = GlobalPoolLayer(bn_post_relu)

        # fully connected layer
        network = DenseLayer(
            avg_pool,
            num_units=10,
            W=HeNormal(),
            nonlinearity=softmax
        )

        return network

    def construct_bottleneck_network(self):
        """Construct the entire resnet with bottleneck blocks."""
        input = InputLayer(
            shape=(None, 3, self.img_dim, self.img_dim),
            input_var=self.input_symbol
        )

        # first layer, output is 16x16x16
        output = batch_norm(ConvLayer(
            input,
            num_filters=16,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=rectify,
            pad='same',
            W=HeNormal(gain='relu')
        ))

        # first stack of residual blocks, output is 64x16x16
        output = self.bottleneck_residual_block(output, first=True)
        for _ in range(1, self.n):
            output = self.bottleneck_residual_bloc(output)

        # second stack of residual blocks, output is 128x8x8
        output = self.bottleneck_residual_bloc(output, increase_dim=True)
        for _ in range(1, self.n):
            output = self.bottleneck_residual_bloc(output)

        # third stack of residual blocks, output is 256x4x4
        output = self.bottleneck_residual_bloc(output, increase_dim=True)
        for _ in range(1, self.n):
            output = self.bottleneck_residual_bloc(output)

        bn_post_conv = BatchNormLayer(output)
        bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

        # average pooling
        avg_pool = GlobalPoolLayer(bn_post_relu)

        # fully connected layer
        network = DenseLayer(
            avg_pool,
            num_units=10,
            W=HeNormal(),
            nonlinearity=softmax
        )

        return network
