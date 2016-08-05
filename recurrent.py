"""RNN Implementations."""

from utils import get_weights, get_bias

import theano
import theano.tensor as T
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
        return_type='all',
        batch_input=False
    ):
        """Initialize weights and biases."""
        # __TODO__ add parameter for number of layers

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_type = return_type
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

        self.weights = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__weights'
        )
        self.recurrent_weights = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__recurrent_weights'
        )

        self.bias = get_bias(output_dim, name=name + '__bias')
        self.h_0 = get_bias(
            output_dim,
            name=name + '__h_0'
        )

        self.h = None

        self.params = [
            self.weights,
            self.recurrent_weights,
            self.bias,
            self.h_0
        ]

    def fprop(self, input):
        """Propogate input forward through the RNN."""
        def recurrence_helper(current_input, recurrent_input):

            return self.activation(
                T.dot(current_input, self.weights) +
                T.dot(recurrent_input, self.recurrent_weights) +
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

        if self.return_type == 'all':
            return self.h
        elif self.return_type == 'last':
            return self.h[-1]
        else:
            raise NotImplementedError("Unknown return type")


class LSTM:
    """Long Short-term Memory Network."""

    def __init__(
        self,
        input_dim,
        output_dim,
        embedding=False,
        name='lstm',
        return_type='all',
        batch_input=False
    ):
        """Initialize weights and biases."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_type = return_type
        self.batch_input = batch_input
        # Intialize forget gate weights
        self.w_fx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_fx'
        )
        self.w_fh = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__w_fh'
        )
        self.w_fc = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__w_fc'
        )

        # Intialize input gate weights
        self.w_ix = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ix'
        )
        self.w_ih = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__w_ih'
        )
        self.w_ic = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__w_ic'
        )

        # Intialize output gate weights
        self.w_ox = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ox'
        )
        self.w_oh = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__w_oh'
        )
        self.w_oc = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__w_oc'
        )

        # Initialize cell weights
        self.w_cx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_cx'
        )
        self.w_ch = get_weights(
            shape=(output_dim, output_dim),
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

        if self.return_type == 'all':
            return self.h
        elif self.return_type == 'last':
            return self.h[-1]
        else:
            raise NotImplementedError("Unknown return type")


class GRU:
    """Gated Recurrent Unit."""

    # http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
    def __init__(
        self,
        input_dim,
        output_dim,
        embedding=False,
        name='gru',
        return_type='all',
        batch_input=False
    ):
        """Initialize weights and biases."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_type = return_type
        self.batch_input = batch_input

        self.w_zx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_fx'
        )
        self.w_zh = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__w_fh'
        )

        self.w_rx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ix'
        )
        self.w_rh = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__w_ih'
        )

        self.w_ox = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ox'
        )
        self.w_oh = get_weights(
            shape=(output_dim, output_dim),
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

        if self.return_type == 'all':
            return self.h
        elif self.return_type == 'last':
            return self.h[-1]
        else:
            raise NotImplementedError("Unknown return type")


class FastLSTM:
    """Faster LSTM by using just 2 weight matrices."""

    def __init__(
        self,
        input_dim,
        output_dim,
        name='lstm',
        return_type='all'
    ):
        """Initialize weights and biases."""
        # __TODO__ add parameter for depth

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_type = return_type

        self.W = get_weights(
            shape=(input_dim, output_dim * 4),
            name=name + '__W'
        )
        self.U = get_weights(
            shape=(output_dim, output_dim * 4),
            name=name + '__U'
        )
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

        if self.return_type == 'all':
            return self.h
        elif self.return_type == 'last':
            return self.h[-1]
        else:
            raise NotImplementedError("Unknown return type")


class MiLSTM:
    """Multiplicative LSTM."""

    def __init__(
        self,
        input_dim,
        output_dim,
        embedding=False,
        name='lstm',
        return_type='all'
    ):
        """Propogate the input through the LSTM."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_type = return_type

        # Intialize block input gates
        self.w_zx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_zx'
        )
        self.w_zh = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_zh'
        )

        # Intialize forget gate weights
        self.w_fx = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_fx'
        )
        self.w_fh = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__w_fh'
        )

        # Intialize input gate weights
        self.w_ix = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ix'
        )
        self.w_ih = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__w_ih'
        )

        # Intialize output gate weights
        self.w_ox = get_weights(
            shape=(input_dim, output_dim),
            name=name + '__w_ox'
        )
        self.w_oh = get_weights(
            shape=(output_dim, output_dim),
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
        def recurrence_helper(x_tm1, h_tm1, z_tm1):
            z_t = T.tanh(
                self.a_z * T.dot(x_tm1, self.w_zx) * T.dot(h_tm1, self.w_zh) +
                self.b_z1 * T.dot(h_tm1, self.w_zh) +
                self.b_z2 * T.dot(x_tm1, self.w_zx) +
                self.b_z
            )
            i_t = T.nnet.sigmoid(
                self.a_i * T.dot(x_tm1, self.w_ix) * T.dot(h_tm1, self.w_ih) +
                self.b_i1 * T.dot(h_tm1, self.w_ih) +
                self.b_i2 * T.dot(x_tm1, self.w_ix) +
                self.b_i
            )
            f_t = T.nnet.sigmoid(
                self.a_f * T.dot(x_tm1, self.w_fx) * T.dot(h_tm1, self.w_fh) +
                self.b_f1 * T.dot(h_tm1, self.w_fh) +
                self.b_f2 * T.dot(x_tm1, self.w_fx) +
                self.b_f
            )
            o_t = T.nnet.sigmoid(
                self.a_o * T.dot(x_tm1, self.w_ox) * T.dot(h_tm1, self.w_oh) +
                self.b_o1 * T.dot(h_tm1, self.w_oh) +
                self.b_o2 * T.dot(x_tm1, self.w_ox) +
                self.b_o
            )
            c_t = i_t * z_t + f_t * z_tm1
            h_t = o_t * T.tanh(c_t)
            return [z_t, h_t]

        outputs_info = [self.z_0, self.h_0]
        [_, self.h], updates = theano.scan(
            fn=recurrence_helper,
            sequences=input,
            outputs_info=outputs_info,
            n_steps=input.shape[0]
        )

        if self.return_type == 'all':
            return self.h
        elif self.return_type == 'last':
            return self.h[-1]
        else:
            raise NotImplementedError("Unknown return type")


class BiRNN:
    """Bidirectional RNN."""

    def __init__(self, forward_rnn, backward_rnn):
        """Initialize forward and backward RNNs."""
        self.forward_rnn = forward_rnn
        self.backward_rnn = backward_rnn

        self.params = self.forward_rnn.params + self.backward_rnn.params

    def fprop(self, input):
        """Progate the input throught the forward and backward RNNS."""
        assert self.forward_rnn.return_type == self.backward_rnn.return_type

        if self.forward_rnn.return_type == 'all':
            return T.concatenate(
                (
                    self.forward_rnn.fprop(input),
                    self.backward_rnn.fprop(input[::-1])
                ),
                axis=1
            )
        elif self.backward_rnn.return_type == 'last':
            return T.concatenate(
                (
                    self.forward_rnn.fprop(input),
                    self.backward_rnn.fprop(input[::-1]))
                )


class BiLSTM:
    """Bidirectional LSTM."""

    def __init__(self, forward_lstm, backward_lstm):
        """Initialize the forward and backward LSTMS."""
        self.forward_lstm = forward_lstm
        self.backward_lstm = backward_lstm

        self.params = self.forward_lstm.params + self.backward_lstm.params

    def fprop(self, input):
        """Propogate the input through the forward and backward LSTMS."""
        assert self.forward_lstm.return_type == self.backward_lstm.return_type

        if self.forward_lstm.return_type == 'all':
            return T.concatenate(
                (
                    self.forward_lstm.fprop(input),
                    self.backward_lstm.fprop(input[::-1])
                ),
                axis=1
            )
        elif self.forward_lstm.return_type == 'last':
            return T.concatenate(
                (
                    self.forward_lstm.fprop(input),
                    self.backward_lstm.fprop(input[::-1]))
                )
