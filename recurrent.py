"""RNN Implementations."""

from utils import get_weights, get_bias, ortho_weight, create_shared

import theano
import theano.tensor as T
import numpy as np
theano.config.floatX = 'float32'

__author__ = "Sandeep Subramanian"
__version__ = "1.0"
__maintainer__ = "Sandeep Subramanian"
__email__ = "sandeep.subramanian@gmail.com"


class RNN:
    """Elman Recurrent Neural Network."""

    def __init__(
        self,
        input_dim,
        output_dim,
        activation='sigmoid',
        embedding=False, name='rnn',
        batch_input=False
    ):
        """Initialize weights and biases."""
        # __TODO__ add parameter for number of layers

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_input = batch_input
        # Set the activation function for this layer
        if activation == 'sigmoid':
            self.activation = T.nnet.sigmoid

        elif activation == 'tanh':
            self.activation = T.tanh

        elif self.activation == 'linear':
            self.activation = None

        else:
            raise NotImplementedError("Unknown activation")

        self.W = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__W'
        )
        self.U = create_shared(
            ortho_weight(output_dim),
            name=name + '__U'
        )

        self.bias = get_bias(output_dim, name=name + '__bias')
        self.h_0 = get_bias(
            output_dim,
            name=name + '__h_0'
        )

        self.h = None

        self.params = [
            self.W,
            self.U,
            self.bias,
            self.h_0
        ]

    def fprop(self, input):
        """Propogate input forward through the RNN."""
        def recurrence_helper(current_input, recurrent_input):

            return self.activation(
                T.dot(current_input, self.W) +
                T.dot(recurrent_input, self.U) +
                self.bias
            )

        if self.batch_input:
            input = input.dimshuffle(1, 0, 2)
            outputs_info = T.alloc(
                self.h_0,
                input.shape[1],
                self.output_dim
            )
        else:
            outputs_info = self.h_0
        self.h, _ = theano.scan(
            fn=recurrence_helper,
            sequences=input,
            outputs_info=outputs_info,
            n_steps=input.shape[0]
        )

        return self.h[-1]


class LSTM:
    """Long Short-term Memory Network."""

    def __init__(
        self,
        input_dim,
        output_dim,
        embedding=False,
        name='lstm',
        batch_input=False
    ):
        """Initialize weights and biases."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_input = batch_input
        # Intialize forget gate weights
        self.w_fx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_fx'
        )
        self.w_fh = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_fh'
        )
        self.w_fc = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_fc'
        )

        # Intialize input gate weights
        self.w_ix = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ix'
        )
        self.w_ih = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_ih'
        )
        self.w_ic = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_ic'
        )

        # Intialize output gate weights
        self.w_ox = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ox'
        )
        self.w_oh = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_oh'
        )
        self.w_oc = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_oc'
        )

        # Initialize cell weights
        self.w_cx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_cx'
        )
        self.w_ch = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_ch'
        )

        # Initialize bias for every gate
        self.b_f = get_bias(output_dim, name=name + '__b_f')
        self.b_i = get_bias(output_dim, name=name + '__b_i')
        self.b_o = get_bias(output_dim, name=name + '__b_o')
        self.b_c = get_bias(output_dim, name=name + '__b_c')

        self.c_0 = get_bias(output_dim, name=name + '__c_0')
        self.h_0 = get_bias(output_dim, name=name + '__h_0')

        self.h = None

        self.params = [
            self.w_fx,
            self.w_fh,
            self.w_fc,
            self.w_ix,
            self.w_ih,
            self.w_ic,
            self.w_ox,
            self.w_oh,
            self.w_oc,
            self.w_cx,
            self.w_ch,
            self.b_f,
            self.b_i,
            self.b_o,
            self.b_c,
            self.c_0,
            self.h_0
        ]

    def fprop(self, input):
        """Propogate the input through the LSTM."""
        def recurrence_helper(x_t, c_tm1, h_tm1):
            i_t = T.nnet.sigmoid(
                T.dot(x_t, self.w_ix) +
                T.dot(h_tm1, self.w_ih) +
                T.dot(c_tm1, self.w_ic) +
                self.b_i
            )
            f_t = T.nnet.sigmoid(
                T.dot(x_t, self.w_fx) +
                T.dot(h_tm1, self.w_fh) +
                T.dot(c_tm1, self.w_fc) +
                self.b_f
            )
            c_t = f_t * c_tm1 + i_t * T.tanh(
                T.dot(x_t, self.w_cx) +
                T.dot(h_tm1, self.w_ch) +
                self.b_c
            )
            o_t = T.nnet.sigmoid(
                T.dot(x_t, self.w_ox) +
                T.dot(h_tm1, self.w_oh) +
                T.dot(c_tm1, self.w_oc) +
                self.b_o
            )
            h_t = o_t * T.tanh(c_t)
            return [c_t, h_t]

        if self.batch_input:
            input = input.dimshuffle(1, 0, 2)
            outputs_info = [
                T.alloc(
                    x,
                    input.shape[1],
                    self.output_dim
                )
                for x in [self.c_0, self.h_0]
            ]
        else:
            outputs_info = [self.c_0, self.h_0]
        [_, self.h], updates = theano.scan(
            fn=recurrence_helper,
            sequences=input,
            outputs_info=outputs_info,
            n_steps=input.shape[0]
        )

        return self.h[-1]


class GRU:
    """Gated Recurrent Unit."""

    # http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
    def __init__(
        self,
        input_dim,
        output_dim,
        embedding=False,
        name='gru',
        batch_input=False
    ):
        """Initialize weights and biases."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_input = batch_input

        self.w_zx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_fx'
        )
        self.w_zh = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_fh'
        )

        self.w_rx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ix'
        )
        self.w_rh = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_rh'
        )

        self.w_ox = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ox'
        )
        self.w_oh = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_oh'
        )

        # Initialize bias for every gate
        self.b_z = get_bias(output_dim, name=name + '__b_f')
        self.b_r = get_bias(output_dim, name=name + '__b_i')
        self.b_o = get_bias(output_dim, name=name + '__b_o')

        self.h_0 = get_bias(output_dim, name=name + '__h_0')

        self.h = None

        self.params = [
            self.w_zx,
            self.w_zh,
            self.w_rx,
            self.w_rh,
            self.w_ox,
            self.w_oh,
            self.b_z,
            self.b_r,
            self.b_o,
            self.h_0
        ]

    def fprop(self, input):
        """Propogate the input through the LSTM."""
        def recurrence_helper(x_t, h_tm1):
            z_t = T.nnet.sigmoid(
                T.dot(x_t, self.w_zx) +
                T.dot(h_tm1, self.w_zh) +
                self.b_z
            )
            r_t = T.nnet.sigmoid(
                T.dot(x_t, self.w_rx) +
                T.dot(h_tm1, self.w_rh) +
                self.b_r
            )
            o_t = T.tanh(
                T.dot(x_t, self.w_ox) +
                T.dot((h_tm1 * r_t), self.w_oh) +
                self.b_o
            )
            h_t = (1 - z_t) * o_t + z_t * h_tm1
            return h_t

        if self.batch_input:
            input = input.dimshuffle(1, 0, 2)
            outputs_info = T.alloc(
                self.h_0,
                input.shape[1],
                self.output_dim
            )
        else:
            outputs_info = self.h_0
        self.h, _ = theano.scan(
            fn=recurrence_helper,
            sequences=input,
            outputs_info=outputs_info,
            n_steps=input.shape[0]
        )

        return self.h[-1]


class FastLSTM:
    """Faster LSTM by using just 2 weight matrices."""

    def __init__(
        self,
        input_dim,
        output_dim,
        batch_input=True,
        name='lstm',
    ):
        """Initialize weights and biases."""
        # __TODO__ add parameter for depth

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_input = batch_input
        if not self.batch_input:
            raise ValueError('Must have batch input as True')

        self.W = get_weights(
            shape=(input_dim, output_dim * 4),
            name=name + '__W'
        )
        self.U = create_shared(np.concatenate([
            ortho_weight(output_dim),
            ortho_weight(output_dim),
            ortho_weight(output_dim),
            ortho_weight(output_dim)
        ], axis=1), name=name + '_U')
        self.b = get_bias(output_dim * 4, name=name + '__b')

        self.c_0 = get_bias(output_dim, name=name + '__c_0')
        self.h_0 = get_bias(output_dim, name=name + '__h_0')

        self.h = None

        self.params = [
            self.W,
            self.U,
            self.b,
            self.c_0,
            self.h_0
        ]

    def _partition_weights(self, matrix, n):
        return matrix[:, n * self.output_dim: (n+1) * self.output_dim]

    def fprop(self, input):
        """Propogate input through the LSTM."""
        def recurrence_helper(x_t, c_tm1, h_tm1):
            p = x_t + T.dot(h_tm1, self.U)
            i_t = T.nnet.sigmoid(
                self._partition_weights(p, 0)
            )
            f_t = T.nnet.sigmoid(
                self._partition_weights(p, 1)
            )
            o_t = T.nnet.sigmoid(
                self._partition_weights(p, 2)
            )
            c_t = T.tanh(
                self._partition_weights(p, 3)
            )
            c_t = f_t * c_tm1 + i_t * c_t
            h_t = o_t * T.tanh(c_t)

            return [c_t, h_t]

        pre_activation = T.dot(input.dimshuffle(1, 0, 2), self.W) + self.b

        outputs_info = [
            T.alloc(x, input.shape[0], self.output_dim) for x in [
                self.c_0, self.h_0
            ]
        ]

        [_, self.h], updates = theano.scan(
            fn=recurrence_helper,
            sequences=pre_activation,
            outputs_info=outputs_info,
            n_steps=input.shape[1]
        )

        return self.h[-1]


class FastGRU:
    """Fast implementation of GRUs."""

    # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
    def __init__(
        self,
        input_dim,
        output_dim,
        batch_input=True,
        name='FastGRU'
    ):
        """Initialize FastGRU parameters."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_input = batch_input

        # Initialize W
        self.W = get_weights(
            shape=(input_dim, 2 * output_dim),
            name=name + '__W'
        )

        self.b = get_bias(
            2 * output_dim,
            name=name + '__b'
        )

        self.U = theano.shared(np.concatenate([
            get_weights(
                shape=(output_dim, output_dim),
                name=name + '__U1',
                strategy='orthogonal',
            ),
            get_weights(
                shape=(output_dim, output_dim),
                name=name + '__U2',
                strategy='orthogonal',
            )
        ], axis=1), borrow=True)

        self.Wx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__Wx'
        )

        self.bx = get_bias(
            output_dim,
            name=name + '__bx'
        )

        self.Ux = theano.shared(get_weights(
            shape=(output_dim, output_dim),
            name=name + '__Ux',
            strategy='orthogonal'
        ), borrow=True)

        self.params = [
            self.W,
            self.U,
            self.b,
            self.Wx,
            self.Ux,
            self.bx
        ]

    def _partition_weights(self, matrix, n):
        if matrix.ndim == 3:
            return matrix[:, :, n * self.output_dim: (n+1) * self.output_dim]
        return matrix[:, n * self.output_dim: (n+1) * self.output_dim]

    def fprop(self, input, input_mask):
        """Propogate input through the network."""
        def recurrence_helper(mask, x_t, xx_t, h_tm1):
            preact = T.dot(h_tm1, self.U)
            preact += x_t

            # reset and update gates
            reset = T.nnet.sigmoid(self._partition_weights(preact, 0))
            update = T.nnet.sigmoid(self._partition_weights(preact, 1))
            preactx = T.dot(h_tm1, self.Ux)
            preactx = preactx * reset
            preactx = preactx + xx_t

            # current hidden state
            h = T.tanh(preactx)

            h = update * h_tm1 + (1. - update) * h
            h = mask[:, None] * h + (1. - mask)[:, None] * h_tm1

            return h

        input = input.dimshuffle(1, 0, 2)
        state_below = T.dot(input, self.W) + self.b
        state_belowx = T.dot(input, self.Wx) + self.bx

        sequences = [input_mask, state_below, state_belowx]
        init_states = [T.alloc(0., input.shape[1], self.output_dim)]
        shared_vars = [self.U, self.Ux]

        self.h, updates = theano.scan(
            fn=recurrence_helper,
            sequences=sequences,
            outputs_info=init_states,
            non_sequences=shared_vars,
            n_steps=input.shape[0],
        )
        return self.h


class MiLSTM:
    """Multiplicative LSTM."""

    # http://arxiv.org/pdf/1606.06630v1.pdf
    def __init__(
        self,
        input_dim,
        output_dim,
        batch_input=True,
        name='lstm',
    ):
        """Initialize MiLSTM parameters."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_input = batch_input

        # Intialize block input gates
        self.w_zx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_zx'
        )
        self.w_zh = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_zh'
        )

        # Intialize forget gate weights
        self.w_fx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_fx'
        )
        self.w_fh = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_fh'
        )

        # Intialize input gate weights
        self.w_ix = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ix'
        )
        self.w_ih = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_ih'
        )

        # Intialize output gate weights
        self.w_ox = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ox'
        )
        self.w_oh = create_shared(
            ortho_weight(output_dim),
            name=name + '__w_oh'
        )

        # Initialize bias for every gate
        self.b_f = get_bias(output_dim, name=name + '__b_f')
        self.b_i = get_bias(output_dim, name=name + '__b_i')
        self.b_o = get_bias(output_dim, name=name + '__b_o')
        self.b_z = get_bias(output_dim, name=name + '__b_z')

        # Bias terms specific to the MiLSTM
        self.b_z1 = get_bias(output_dim, name=name + '__b_z1')
        self.b_z2 = get_bias(output_dim, name=name + '__b_z2')
        self.a_z = get_bias(output_dim, name=name + '__a_z')

        self.b_i1 = get_bias(output_dim, name=name + '__b_i1')
        self.b_i2 = get_bias(output_dim, name=name + '__b_i2')
        self.a_i = get_bias(output_dim, name=name + '__a_i')

        self.b_f1 = get_bias(output_dim, name=name + '__b_f1')
        self.b_f2 = get_bias(output_dim, name=name + '__b_f2')
        self.a_f = get_bias(output_dim, name=name + '__a_f')

        self.b_o1 = get_bias(output_dim, name=name + '__b_o1')
        self.b_o2 = get_bias(output_dim, name=name + '__b_o2')
        self.a_o = get_bias(output_dim, name=name + '__a_o')

        self.z_0 = get_bias(output_dim, name=name + '__z_0')
        self.h_0 = get_bias(output_dim, name=name + '__h_0')

        self.h = None

        self.params = [
            self.w_fx,
            self.w_fh,
            self.w_ix,
            self.w_ih,
            self.w_ox,
            self.w_oh,
            self.b_f,
            self.b_i,
            self.b_o,
            self.z_0,
            self.h_0
        ]
        self.params.extend([
            self.b_z1,
            self.b_z2,
            self.a_z,
            self.b_i1,
            self.b_i2,
            self.a_i,
            self.b_f1,
            self.b_f2,
            self.a_f,
            self.b_o1,
            self.b_o2,
            self.a_o
        ])

    def fprop(self, input):
        """Propogate the input through the LSTM."""
        def recurrence_helper(x_t, z_tm1, h_tm1):
            z_t = T.tanh(
                self.a_z * T.dot(x_t, self.w_zx) * T.dot(h_tm1, self.w_zh) +
                self.b_z1 * T.dot(h_tm1, self.w_zh) +
                self.b_z2 * T.dot(x_t, self.w_zx) +
                self.b_z
            )
            i_t = T.nnet.sigmoid(
                self.a_i * T.dot(x_t, self.w_ix) * T.dot(h_tm1, self.w_ih) +
                self.b_i1 * T.dot(h_tm1, self.w_ih) +
                self.b_i2 * T.dot(x_t, self.w_ix) +
                self.b_i
            )
            f_t = T.nnet.sigmoid(
                self.a_f * T.dot(x_t, self.w_fx) * T.dot(h_tm1, self.w_fh) +
                self.b_f1 * T.dot(h_tm1, self.w_fh) +
                self.b_f2 * T.dot(x_t, self.w_fx) +
                self.b_f
            )
            c_t = i_t * z_t + f_t * z_tm1
            o_t = T.nnet.sigmoid(
                self.a_o * T.dot(x_t, self.w_ox) * T.dot(h_tm1, self.w_oh) +
                self.b_o1 * T.dot(h_tm1, self.w_oh) +
                self.b_o2 * T.dot(x_t, self.w_ox) +
                self.b_o
            )
            h_t = o_t * T.tanh(c_t)
            return [z_t, h_t]

        if self.batch_input:
            input = input.dimshuffle(1, 0, 2)
            outputs_info = [
                T.alloc(
                    x,
                    input.shape[1],
                    self.output_dim
                )
                for x in [self.z_0, self.h_0]
            ]
        else:
            outputs_info = [self.z_0, self.h_0]
        [_, self.h], updates = theano.scan(
            fn=recurrence_helper,
            sequences=input,
            outputs_info=outputs_info,
            n_steps=input.shape[0]
        )

        return self.h[-1]


class BiRNN:
    """Bidirectional RNN."""

    # Forward and Backward RNNs can be any of the above RNN types.
    def __init__(self, forward_rnn, backward_rnn):
        """Initialize forward and backward RNNs."""
        self.forward_rnn = forward_rnn
        self.backward_rnn = backward_rnn

        self.params = self.forward_rnn.params + self.backward_rnn.params

    def fprop(self, input):
        """Progate the input throught the forward and backward RNNS."""
        f_h = self.forward_rnn.fprop(input)
        self.h = T.concatenate(
            (
                self.forward_rnn.h,
                self.backward_rnn.h[::-1]
            ),
            axis=1
        )

        return T.concatenate((f_h, self.backward_rnn.h[-1]))


class MultiLayerRNN:
    """Multi-layer RNN."""

    def __init__(
        self,
        num_layers,
        cell_type,
        input_dim,
        output_dim,
        batch_input
    ):
        """Initialize the list of RNNs."""
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_input = batch_input
        self.rnns = []

        cell_types = {
            'FastLSTM': FastLSTM,
            'GRU': GRU,
            'MiLSTM': MiLSTM,
            'LSTM': LSTM
        }

        self.rnn_cell = cell_types[self.cell_type]

        for i in xrange(self.num_layers):
            if i == 0:
                self.rnns.append(
                    self.rnn_cell(
                        self.input_dim,
                        self.output_dim,
                        self.batch_input,
                        name='rnn_layer_%d' % (i)
                    )
                )
            else:
                self.rnns.append(
                    self.rnn_cell(
                        self.output_dim,
                        self.output_dim,
                        self.batch_input,
                        name='rnn_layer_%d' % (i)
                    )
                )

    def fprop(self, input):
        """Propogate the input through the network."""
        output = input
        for rnn in self.rnns:
            rnn.fprop(output)
            output = rnn.h
        return output


class LNFastLSTM:
    """Layer normalized FastLSTM."""

    def __init__(
        self,
        input_dim,
        output_dim,
        batch_input=True,
        name='lstm',
    ):
        """Initialize weights and biases."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_input = batch_input
        if not self.batch_input:
            raise ValueError('Must have batch input as True')

        self.W = get_weights(
            shape=(input_dim, output_dim * 4),
            name=name + '__W'
        )
        self.U = create_shared(np.concatenate([
            ortho_weight(output_dim),
            ortho_weight(output_dim),
            ortho_weight(output_dim),
            ortho_weight(output_dim)
        ], axis=1), name=name + '_U')
        self.b = get_bias(output_dim * 4, name=name + '__b')

        self.c_0 = get_bias(output_dim, name=name + '__c_0')
        self.h_0 = get_bias(output_dim, name=name + '__h_0')

        self.h = None

        # Layer Normalization scale and shift params
        self.scale_add = 0.
        self.scale_mul = 1.
        self.b1 = create_shared(self.scale_add * np.ones(
            (4 * self.output_dim)
        ).astype('float32'), name=name + '__b1')
        self.b2 = create_shared(self.scale_add * np.ones(
            (4 * self.output_dim)
        ).astype('float32'), name=name + '__b2')
        self.b3 = create_shared(self.scale_add * np.ones(
            (1 * self.output_dim)
        ).astype('float32'), name=name + '__b3')

        self.s1 = create_shared(self.scale_mul * np.ones(
            (4 * self.output_dim)
        ).astype('float32'), name=name + '__s1')
        self.s2 = create_shared(self.scale_mul * np.ones(
            (4 * self.output_dim)
        ).astype('float32'), name=name + '__s2')
        self.s3 = create_shared(self.scale_mul * np.ones(
            (1 * self.output_dim)
        ).astype('float32'), name=name + '__s3')

        self.params = [
            self.W,
            self.U,
            self.b,
            self.c_0,
            self.b1,
            self.s1,
            self.b2,
            self.s2,
            self.b3,
            self.s3,
            self.h_0
        ]

    def _layer_norm(self, x, b, s):
        self._eps = 1e-5
        output = (x - x.mean(1)[:, None]) / T.sqrt(
            (x.var(1)[:, None] + self._eps)
        )
        output = s[None, :] * output + b[None, :]
        return output

    def _partition_weights(self, matrix, n):
        return matrix[:, n * self.output_dim: (n+1) * self.output_dim]

    def fprop(self, input):
        """Propogate input through the LSTM."""
        def recurrence_helper(x_t, c_tm1, h_tm1):
            x_t = self._layer_norm(x_t, self.b1, self.s1)
            p = self._layer_norm(x_t + T.dot(h_tm1, self.U), self.b2, self.s2)
            i_t = T.nnet.sigmoid(
                self._partition_weights(p, 0)
            )
            f_t = T.nnet.sigmoid(
                self._partition_weights(p, 1)
            )
            o_t = T.nnet.sigmoid(
                self._partition_weights(p, 2)
            )
            c_t = T.tanh(
                self._partition_weights(p, 3)
            )
            c_t = self._layer_norm(f_t * c_tm1 + i_t * c_t, self.b3, self.s3)
            h_t = o_t * T.tanh(c_t)

            return [c_t, h_t]

        pre_activation = T.dot(input.dimshuffle(1, 0, 2), self.W) + self.b

        outputs_info = [
            T.alloc(x, input.shape[0], self.output_dim) for x in [
                self.c_0, self.h_0
            ]
        ]

        [_, self.h], updates = theano.scan(
            fn=recurrence_helper,
            sequences=pre_activation,
            outputs_info=outputs_info,
            n_steps=input.shape[1]
        )

        return self.h


class LNFastGRU:
    """Fast implementation of GRUs."""

    # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
    def __init__(
        self,
        input_dim,
        output_dim,
        batch_input=True,
        name='FastGRU'
    ):
        """Initialize FastGRU parameters."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_input = batch_input

        # Initialize W
        self.W = get_weights(
            shape=(input_dim, 2 * output_dim),
            name=name + '__W'
        )

        self.b = get_bias(
            2 * output_dim,
            name=name + '__b'
        )

        self.U = theano.shared(np.concatenate([
            get_weights(
                shape=(output_dim, output_dim),
                name=name + '__U1',
                strategy='orthogonal',
            ),
            get_weights(
                shape=(output_dim, output_dim),
                name=name + '__U2',
                strategy='orthogonal',
            )
        ], axis=1), borrow=True)

        self.Wx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__Wx'
        )

        self.bx = get_bias(
            output_dim,
            name=name + '__bx'
        )

        self.Ux = theano.shared(get_weights(
            shape=(output_dim, output_dim),
            name=name + '__Ux',
            strategy='orthogonal'
        ), borrow=True)

        # Layer Normalization scale and shift params
        self.scale_add = 0.
        self.scale_mul = 1.
        self.b1 = create_shared(self.scale_add * np.ones(
            (4 * self.output_dim)
        ).astype('float32'), name=name + '__b1')
        self.b2 = create_shared(self.scale_add * np.ones(
            (4 * self.output_dim)
        ).astype('float32'), name=name + '__b2')
        self.b3 = create_shared(self.scale_add * np.ones(
            (1 * self.output_dim)
        ).astype('float32'), name=name + '__b3')
        self.b4 = create_shared(self.scale_add * np.ones(
            (1 * self.output_dim)
        ).astype('float32'), name=name + '__b4')
        self.s1 = create_shared(self.scale_mul * np.ones(
            (4 * self.output_dim)
        ).astype('float32'), name=name + '__s1')
        self.s2 = create_shared(self.scale_mul * np.ones(
            (4 * self.output_dim)
        ).astype('float32'), name=name + '__s2')
        self.s3 = create_shared(self.scale_mul * np.ones(
            (1 * self.output_dim)
        ).astype('float32'), name=name + '__s3')
        self.s4 = create_shared(self.scale_mul * np.ones(
            (1 * self.output_dim)
        ).astype('float32'), name=name + '__s4')

        self.params = [
            self.W,
            self.U,
            self.b,
            self.Wx,
            self.Ux,
            self.bx,
            self.b1,
            self.s1,
            self.b2,
            self.s2,
            self.b3,
            self.s3,
            self.b4,
            self.s4
        ]

    def _partition_weights(self, matrix, n):
        if matrix.ndim == 3:
            return matrix[:, :, n * self.output_dim: (n + 1) * self.output_dim]
        return matrix[:, n * self.output_dim: (n + 1) * self.output_dim]

    def _layer_norm(self, x, b, s):
        self._eps = 1e-5
        output = (x - x.mean(1)[:, None]) / T.sqrt(
            (x.var(1)[:, None] + self._eps)
        )
        output = s[None, :] * output + b[None, :]
        return output

    def fprop(self, input, input_mask):
        """Propogate input through the network."""
        def recurrence_helper(mask, x_t, xx_t, h_tm1):
            preact = self._layer_norm(T.dot(h_tm1, self.U), self.b3, self.s3)
            preact += x_t

            # reset and update gates
            reset = T.nnet.sigmoid(self._partition_weights(preact, 0))
            update = T.nnet.sigmoid(self._partition_weights(preact, 1))
            preactx = self._layer_norm(T.dot(h_tm1, self.Ux), self.b4, self.s4)
            preactx = preactx * reset
            preactx = preactx + xx_t

            # current hidden state
            h = T.tanh(preactx)

            h = update * h_tm1 + (1. - update) * h
            h = mask[:, None] * h + (1. - mask)[:, None] * h_tm1

            return h

        input = input.dimshuffle(1, 0, 2)
        state_below = self._layer_norm(T.dot(input, self.W) + self.b1, self.s1)
        state_belowx = self._layer_norm(T.dot(input, self.Wx) + self.bx, self.b2, self.s2)

        sequences = [input_mask, state_below, state_belowx]
        init_states = [T.alloc(0., input.shape[1], self.output_dim)]
        shared_vars = [self.U, self.Ux]

        self.h, updates = theano.scan(
            fn=recurrence_helper,
            sequences=sequences,
            outputs_info=init_states,
            non_sequences=shared_vars,
            n_steps=input.shape[0],
        )
        return self.h
