"""Attenion Mechanisms."""
import theano
import theano.tensor as T
from utils import get_weights, get_bias
import numpy as np


class FastLSTMAttention:
    """Faster LSTM by using just 2 weight matrices."""

    def __init__(
        self,
        input_dim,
        output_dim,
        ctx_dim,
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
        self.U = get_weights(
            shape=(output_dim, output_dim * 4),
            name=name + '__U'
        )
        self.v = get_weights(
            shape=(output_dim, 1),
            name=name + '__v'
        )
        self.Wc = get_weights(
            shape=(ctx_dim, output_dim),
            name=name + '__Wc'
        )
        self.Wh = get_weights(
            shape=(output_dim, output_dim),
            name=name + '__Wh'
        )
        self.Watt = get_weights(
            shape=(2 * output_dim, output_dim),
            name=name + '__Watt'
        )
        self.batt = get_bias(output_dim, name=name + '__batt')
        self.b = get_bias(output_dim * 4, name=name + '__b')

        self.c_0 = get_bias(output_dim, name=name + '__c_0')
        self.h_0 = get_bias(output_dim, name=name + '__h_0')

        self.h = None

        self.params = [
            self.W,
            self.U,
            self.b,
            self.c_0,
            self.v,
            self.Wc,
            self.Wh,
            self.Watt,
            self.batt,
            self.h_0
        ]

    def _partition_weights(self, matrix, n):
        return matrix[:, n * self.output_dim: (n+1) * self.output_dim]

    def fprop(self, input, ctx):
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

            u_t = T.dot(ctx, self.Wc) + \
                T.dot(h_t.dimshuffle(0, 'x', 1), self.Wh)
            u_t = T.tanh(u_t).dot(self.v)
            att = T.nnet.softmax(u_t.reshape((input.shape[0], ctx.shape[1])))
            att = T.sum(att.dimshuffle(0, 1, 'x') * ctx, axis=1)
            h_t = T.concatenate((h_t, att), axis=-1)
            h_t = T.tanh(T.dot(h_t, self.Watt) + self.batt)
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


class FastGRUAttention:
    """Fast implementation of Attention GRUs."""

    # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
    def __init__(
        self,
        input_dim,
        output_dim,
        ctx_dim,
        batch_input=True,
        name='FastGRU'
    ):
        """Initialize FastGRU parameters."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ctx_dim = ctx_dim
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
            strategy='orthogonal',
            name=name + '__Ux',
        ), borrow=True)

        self.U_nl = theano.shared(np.concatenate([
            get_weights(
                shape=(output_dim, output_dim),
                name=name + '__U_nl1',
                strategy='orthogonal',
            ),
            get_weights(
                shape=(output_dim, output_dim),
                name=name + '__U_nl2',
                strategy='orthogonal',
            )
        ], axis=1), borrow=True)

        self.b_nl = get_bias(
            2 * output_dim,
            name=name + '__bnl'
        )

        self.Ux_nl = theano.shared(get_weights(
            shape=(output_dim, output_dim),
            name=name + '__Ux_nl',
            strategy='orthogonal'
        ), borrow=True)

        self.bx_nl = get_bias(
            output_dim,
            name=name + '__bx_nl'
        )

        self.Wc = get_weights(
            shape=(ctx_dim, output_dim * 2),
            name=name + '__Wc'
        )

        self.Wcx = get_weights(
            shape=(ctx_dim, output_dim),
            name=name + '__Wcx'
        )

        self.W_comb_att = get_weights(
            shape=(output_dim, ctx_dim),
            name=name + '__W_comb_att'
        )

        self.Wc_att = get_weights(
            shape=(ctx_dim, ctx_dim),
            name=name + '__Wc_att'
        )

        self.b_att = get_bias(
            ctx_dim,
            name=name + '__b_att'
        )

        self.U_att = get_weights(
            shape=(ctx_dim, 1),
            name=name + '__U_att'
        )

        self.c_att = get_bias(
            1,
            name=name + '__c_att'
        )

        self.h_0 = None

        self.params = [
            self.W,
            self.U,
            self.b,
            self.Wx,
            self.Ux,
            self.bx,
            self.U_nl,
            self.b_nl,
            self.Ux_nl,
            self.bx_nl,
            self.Wc,
            self.Wcx,
            self.W_comb_att,
            self.Wc_att,
            self.b_att,
            self.U_att,
            self.c_att
        ]

    def _partition_weights(self, matrix, n):
        if matrix.ndim == 3:
            return matrix[:, :, n * self.output_dim: (n+1) * self.output_dim]
        return matrix[:, n * self.output_dim: (n+1) * self.output_dim]

    def fprop(self, input, context, context_mask, tgt_mask):
        """Propogate input through the network."""
        def recurrence_helper(mask, x_t, xx_t, h_tm1, ctx_, alpha, pctx_, cc_):
            preact1 = T.dot(h_tm1, self.U)
            preact1 += x_t
            preact1 = T.nnet.sigmoid(preact1)

            # reset and update gates
            reset1 = self._partition_weights(preact1, 0)
            update1 = self._partition_weights(preact1, 1)

            preactx1 = T.dot(h_tm1, self.Ux)
            preactx1 *= reset1
            preactx1 += xx_t

            # current hidden state
            h1 = T.tanh(preactx1)

            h1 = update1 * h_tm1 + (1. - update1) * h1
            h1 = mask[:, None] * h1 + (1. - mask)[:, None] * h_tm1

            # Attention model
            pstate_ = T.dot(h1, self.W_comb_att)
            pctx__ = pctx_ + pstate_[None, :, :]
            pctx__ = T.tanh(pctx__)
            alpha = T.dot(pctx__, self.U_att) + self.c_att
            alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
            alpha = T.exp(alpha)
            if context_mask:
                alpha = alpha * context_mask
            alpha = alpha / alpha.sum(0, keepdims=True)
            ctx_ = (cc_ * alpha[:, :, None]).sum(0)

            preact2 = T.dot(h1, self.U_nl) + self.b_nl
            preact2 += T.dot(ctx_, self.Wc)
            preact2 = T.nnet.sigmoid(preact2)

            reset2 = self._partition_weights(preact2, 0)
            update2 = self._partition_weights(preact2, 1)

            preactx2 = T.dot(h1, self.Ux_nl) + self.bx_nl
            preactx2 *= reset2
            preactx2 += T.dot(ctx_, self.Wcx)

            h2 = T.tanh(preactx2)
            h2 = update2 * h1 + (1. - update2) * h2
            h2 = mask[:, None] * h2 + (1. - mask)[:, None] * h1

            return h2, ctx_, alpha.T

        input = input.dimshuffle(1, 0, 2)
        state_below = T.dot(input, self.W) + self.b
        state_belowx = T.dot(input, self.Wx) + self.bx

        pctx_ = T.dot(context, self.Wc_att) + self.b_att

        sequences = [tgt_mask, state_below, state_belowx]
        if self.h_0 is None:
            self.h_0 = T.alloc(0., input.shape[1], self.output_dim)

        [self.h, self.ctx_, self.alpha], updates = theano.scan(
            fn=recurrence_helper,
            sequences=sequences,
            outputs_info=[
                self.h_0,
                T.alloc(0., input.shape[1], context.shape[2]),
                T.alloc(0., input.shape[1], context.shape[0]),
            ],
            non_sequences=[pctx_, context],
            n_steps=input.shape[0],
        )

        return self.h
