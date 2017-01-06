import theano
import theano.tensor as T

from utils import create_shared
from layer import FullyConnectedLayer
import numpy as np


class CRF(object):
    """Conditional Random Field."""

    def __init__(self, n_tags):
        """Initialize CRF params."""
        self.n_tags = n_tags

        self.transition_matrix = np.random.rand(self.n_tags + 2, self.n_tags + 2)
        self.transition_matrix = create_shared(
            self.transition_matrix.astype(np.float32),
            name='A'
        )

        self.eps = -1000
        self.b_s = np.array(
            [[self.eps] * self.n_tags + [0, self.eps]]
        ).astype(np.float32)
        self.e_s = np.array(
            [[self.eps] * self.n_tags + [self.eps, 0]]
        ).astype(np.float32)

        self.b_id = theano.shared(
            value=np.array([self.n_tags], dtype=np.int32)
        )
        self.e_id = theano.shared(
            value=np.array([self.n_tags + 1], dtype=np.int32)
        )

        self.params = [self.transition_matrix]

    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        """
        Takes as input:
            - observations, sequence of shape (n_steps, n_classes)
            - transitions, sequence of shape (n_classes, n_classes)
        Probabilities must be given in the log space.
        Compute alpha, matrix of size (n_steps, n_classes), such that
        alpha[i, j] represents one of these 2 values:
            - the probability that the real path at node i ends in j
            - the maximum probability of a path finishing in j at node i (V)
        Returns one of these 2 values:
            - alpha
            - the final probability, which can be:
                - the sum of the probabilities of all paths
                - the probability of the best path (Viterbi)
        """
        assert not return_best_sequence or (viterbi and not return_alpha)

        def recurrence(obs, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            if viterbi:
                scores = previous + obs + self.transition_matrix
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)
                    return out, out2
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + self.transition_matrix,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=(initial, None) if return_best_sequence else initial,
            sequences=[observations[1:]],
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(
                fn=lambda beta_i, previous: beta_i[previous],
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32'),
                sequences=T.cast(self.alpha[1][::-1], 'int32')
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input.shape[0]
        observations = T.concatenate(
            [input, self.eps * T.ones((seq_length, 2))],
            axis=1
        )
        observations = T.concatenate(
            [self.b_s, observations, self.e_s],
            axis=0
        )
        if mode != 'train':
            return self.alpha_recursion(
                observations,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score = input[T.arange(seq_length), ground_truth].sum()
        padded_tags_ids = T.concatenate(
            [self.b_id, ground_truth, self.e_id],
            axis=0
        )
        real_path_score += self.transition_matrix[
            padded_tags_ids[T.arange(seq_length + 1)],
            padded_tags_ids[T.arange(seq_length + 1) + 1]
        ].sum()

        all_paths_scores = self.alpha_recursion(
            observations,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        cost = - (real_path_score - all_paths_scores)
        return cost


class BatchCRF(object):
    """Minibatch Conditional Random Field."""

    def __init__(self, n_tags):
        """Initialize CRF params."""
        self.n_tags = n_tags
        self.transition_matrix = np.random.rand(self.n_tags + 2, self.n_tags + 2)
        self.transition_matrix = create_shared(
            self.transition_matrix.astype(np.float32),
            name='A'
        )

        self.eps = -1000
        self.b_s = theano.shared(np.array(
            [[self.eps] * self.n_tags + [0, self.eps]]
        ).astype(np.float32), borrow=True)
        self.e_s = theano.shared(np.array(
            [[self.eps] * self.n_tags + [self.eps, 0]]
        ).astype(np.float32), borrow=True)

        self.b_id = theano.shared(
            value=np.array([self.n_tags], dtype=np.int32)
        )
        self.e_id = theano.shared(
            value=np.array([self.n_tags + 1], dtype=np.int32)
        )

        self.params = [self.transition_matrix]

    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        """
        Takes as input:
            - observations, sequence of shape (n_steps, n_batch, n_classes)
            - transitions, sequence of shape (n_classes, n_batch, n_classes)
        Probabilities must be given in the log space.
        Compute alpha, matrix of size (n_steps, n_batch, n_classes), such that
        alpha[i, :, j] represents one of these 2 values:
            - the probability that the real path at node i ends in j
            - the maximum probability of a path finishing in j at node i (V)
        Returns one of these 2 values:
            - alpha
            - the final probability, which can be:
                - the sum of the probabilities of all paths
                - the probability of the best path (Viterbi)
        """
        assert not return_best_sequence or (viterbi and not return_alpha)

        def recurrence(obs, previous):
            previous = previous.dimshuffle(0, 1, 'x')
            obs = obs.dimshuffle(0, 'x', 1)
            if viterbi:
                scores = previous + obs + self.transition_matrix
                out = scores.max(axis=1)
                if return_best_sequence:
                    out2 = scores.argmax(axis=1)
                    return out, out2
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + self.transition_matrix,
                    axis=1
                )

        def sequence_function(beta_i, previous):
            return beta_i[T.arange(previous.shape[0]), previous]

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=(initial, None) if return_best_sequence else initial,
            sequences=[observations[1:]],
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(
                fn=sequence_function,
                outputs_info=T.cast(T.argmax(self.alpha[0][-1], axis=1), 'int32'),
                sequences=T.cast(self.alpha[1][::-1], 'int32')
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1], axis=1)]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1, :], axis=1)

    def fprop(
        self,
        input,
        ground_truth,
        mask,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        input = input.dimshuffle(1, 0, 2)
        batch_size = input.shape[1]
        seq_length = input.shape[0]
        pad_item = self.eps * T.ones((seq_length, 2))
        pad_item = T.repeat(pad_item.dimshuffle(0, 'x', 1), batch_size, axis=1)
        observations = T.concatenate(
            [input, pad_item],
            axis=-1
        )
        b_s = T.repeat(self.b_s.dimshuffle(0, 'x', 1), batch_size, axis=1)
        e_s = T.repeat(self.e_s.dimshuffle(0, 'x', 1), batch_size, axis=1)
        observations = T.concatenate(
            [b_s, observations, e_s],
            axis=0
        )
        if mode != 'train':
            return self.alpha_recursion(
                observations,
                viterbi,
                return_alpha,
                return_best_sequence
            )

        real_path_score = input.dimshuffle(1, 0, 2)[:, T.arange(seq_length), ground_truth]
        real_path_score = real_path_score[T.arange(batch_size), T.arange(batch_size)]
        real_path_score = real_path_score * mask
        real_path_score = real_path_score.sum(axis=1).mean()
        
        b_id = T.repeat(self.b_id.dimshuffle('x', 0), batch_size, axis=0)
        e_id = T.repeat(self.e_id.dimshuffle('x', 0), batch_size, axis=0)

        padded_tags_ids = T.concatenate(
            [b_id, ground_truth, e_id],
            axis=1
        )

        real_path_score += (self.transition_matrix[
            padded_tags_ids[:, T.arange(seq_length + 1)],
            padded_tags_ids[:, T.arange(seq_length + 1) + 1]
        ] * T.concatenate([mask, T.zeros((batch_size, 1))], axis=-1)).sum(axis=1).mean()

        masked_observations = observations.dimshuffle(1, 0, 2) * T.concatenate([
            T.zeros((batch_size, 1)),
            mask,
            T.zeros((batch_size, 1))
        ], axis=1).dimshuffle(0, 1, 'x')
        all_paths_scores = self.alpha_recursion(
            masked_observations.dimshuffle(1, 0, 2),
            viterbi,
            return_alpha,
            return_best_sequence
        ).mean()
        cost = - (real_path_score - all_paths_scores)
        return cost


class BatchCRFHiddenPotential(object):
    """Minibatch Conditional Random Field."""

    def __init__(self, n_tags, hidden_dim):
        """Initialize CRF params."""
        self.n_tags = n_tags
        self.hidden_dim = hidden_dim

        self.transition_matrix = np.random.rand(self.n_tags + 2, self.n_tags + 2)
        self.transition_matrix = create_shared(
            self.transition_matrix.astype(np.float32),
            name='A'
        )
        self.fc_o = FullyConnectedLayer(
            input_dim=self.n_tags + 2,
            output_dim=self.n_tags + 2,
            activation='tanh'
        )
        self.fc_h = FullyConnectedLayer(
            input_dim=self.hidden_dim,
            output_dim=self.n_tags + 2,
            activation='tanh'
        )

        self.eps = -1000
        self.b_s = theano.shared(np.array(
            [[self.eps] * self.n_tags + [0, self.eps]]
        ).astype(np.float32), borrow=True)
        self.e_s = theano.shared(np.array(
            [[self.eps] * self.n_tags + [self.eps, 0]]
        ).astype(np.float32), borrow=True)

        self.b_id = theano.shared(
            value=np.array([self.n_tags], dtype=np.int32)
        )
        self.e_id = theano.shared(
            value=np.array([self.n_tags + 1], dtype=np.int32)
        )

        self.params = [self.transition_matrix, self.fc_o, self.fc_h]

    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        hidden_states,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        """
        Takes as input:
            - observations, sequence of shape (n_steps, n_batch, n_classes)
            - transitions, sequence of shape (n_classes, n_batch, n_classes)
        Probabilities must be given in the log space.
        Compute alpha, matrix of size (n_steps, n_batch, n_classes), such that
        alpha[i, :, j] represents one of these 2 values:
            - the probability that the real path at node i ends in j
            - the maximum probability of a path finishing in j at node i (V)
        Returns one of these 2 values:
            - alpha
            - the final probability, which can be:
                - the sum of the probabilities of all paths
                - the probability of the best path (Viterbi)
        """
        assert not return_best_sequence or (viterbi and not return_alpha)

        def recurrence(obs, h_t, previous, h_tm1):
            obs_potential = self.fc_o.fprop(obs)
            previous_potential = self.fc_o.fprop(previous)

            h_t_potential = self.fc_h.fprop(h_t)
            h_tm1_potential = self.fc_h.fprop(h_tm1)

            overall_potential = obs_potential.dimshuffle(0, 'x', 1) + previous_potential.dimshuffle(0, 1, 'x') + \
                h_t_potential.dimshuffle(0, 'x', 1) + h_tm1_potential.dimshuffle(0, 1, 'x')

            previous = previous.dimshuffle(0, 1, 'x')
            obs = obs.dimshuffle(0, 'x', 1)
            if viterbi:
                scores = previous + obs + self.transition_matrix
                out = scores.max(axis=1)
                if return_best_sequence:
                    out2 = scores.argmax(axis=1)
                    return out, out2
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + self.transition_matrix + overall_potential,
                    axis=1
                ), h_t

        def sequence_function(beta_i, previous):
            return beta_i[T.arange(previous.shape[0]), previous]

        initial = observations[0]
        initial_h = hidden_states[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=(initial, initial_h, None) if return_best_sequence else (initial, initial_h),
            sequences=[observations[1:], hidden_states[1:]],
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(
                fn=sequence_function,
                outputs_info=T.cast(T.argmax(self.alpha[0][-1], axis=1), 'int32'),
                sequences=T.cast(self.alpha[2][::-1], 'int32')
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1], axis=1)]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[0][-1, :], axis=1)

    def fprop(
        self,
        input,
        ground_truth,
        hidden_states,
        mask,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        input = input.dimshuffle(1, 0, 2)
        batch_size = input.shape[1]
        seq_length = input.shape[0]
        pad_item = self.eps * T.ones((seq_length, 2))
        pad_item = T.repeat(pad_item.dimshuffle(0, 'x', 1), batch_size, axis=1)
        observations = T.concatenate(
            [input, pad_item],
            axis=-1
        )
        b_s = T.repeat(self.b_s.dimshuffle(0, 'x', 1), batch_size, axis=1)
        e_s = T.repeat(self.e_s.dimshuffle(0, 'x', 1), batch_size, axis=1)
        observations = T.concatenate(
            [b_s, observations, e_s],
            axis=0
        )
        self.b_h = T.zeros((batch_size, self.hidden_dim)).dimshuffle('x', 0, 1)
        self.e_h = T.zeros((batch_size, self.hidden_dim)).dimshuffle('x', 0, 1)
        hidden_states = T.concatenate(
            [self.b_h, hidden_states * mask.dimshuffle(1, 0, 'x'), self.e_h],
            axis=0
        )
        if mode != 'train':
            return self.alpha_recursion(
                observations,
                viterbi,
                return_alpha,
                return_best_sequence
            )

        real_path_score = input.dimshuffle(1, 0, 2)[:, T.arange(seq_length), ground_truth]
        real_path_score = real_path_score[T.arange(batch_size), T.arange(batch_size)]
        real_path_score = real_path_score * mask
        real_path_score = real_path_score.sum(axis=1).mean()

        b_id = T.repeat(self.b_id.dimshuffle('x', 0), batch_size, axis=0)
        e_id = T.repeat(self.e_id.dimshuffle('x', 0), batch_size, axis=0)

        padded_tags_ids = T.concatenate(
            [b_id, ground_truth, e_id],
            axis=1
        )

        real_path_score += (self.transition_matrix[
            padded_tags_ids[:, T.arange(seq_length + 1)],
            padded_tags_ids[:, T.arange(seq_length + 1) + 1]
        ] * T.concatenate([mask, T.zeros((batch_size, 1))], axis=-1)).sum(axis=1).mean()

        masked_observations = observations.dimshuffle(1, 0, 2) * T.concatenate([
            T.zeros((batch_size, 1)),
            mask,
            T.zeros((batch_size, 1))
        ], axis=1).dimshuffle(0, 1, 'x')
        all_paths_scores = self.alpha_recursion(
            masked_observations.dimshuffle(1, 0, 2),
            hidden_states,
            viterbi,
            return_alpha,
            return_best_sequence
        ).mean()
        cost = - (real_path_score - all_paths_scores)
        return cost


class BatchCRFForwardBackward(object):
    """Minibatch Conditional Random Field."""

    def __init__(self, input_dim):
        """Initialize CRF params."""
        self.input_dim = input_dim
        self.transition_matrix = np.random.rand(self.input_dim, self.input_dim)
        self.transition_matrix = create_shared(
            self.transition_matrix.astype(np.float32),
            name='A'
        )
        self.initial = np.zeros((1, self.input_dim))
        self.initial = create_shared(
            self.initial.astype(np.float32),
            name='initial'
        )

        self.params = [self.transition_matrix, self.initial]

    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def _forward(
        self,
        observations,
        mask,
        transition_matrix
    ):
        def recurrence(obs, mask, previous, transition_matrix):
            previous = previous.dimshuffle(0, 1, 'x')
            obs = obs.dimshuffle(0, 'x', 1)
            return self.log_sum_exp(
                previous + obs + transition_matrix.dimshuffle('x', 0, 1),
                axis=1
            ) * mask.dimshuffle(0, 'x')

        initial = observations[0].dimshuffle(0, 'x', 1)
        transition_init = self.initial.dimshuffle('x', 0, 1)
        T.addbroadcast(transition_init, 1)
        first_input = initial + T.zeros_like(initial.dimshuffle(0, 2, 1)) + \
            transition_init
        initial = self.log_sum_exp(first_input, axis=1) * \
            mask[0].dimshuffle(0, 'x')

        alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=[initial],
            sequences=[observations[1:], mask[1:]],
            non_sequences=[transition_matrix]
        )

        return T.concatenate([initial.dimshuffle('x', 0, 1), alpha])

    def _backward(
        self,
        observations,
        mask,
        transition_matrix
    ):
        def recurrence(obs, mask, previous, transition_matrix):
            previous = previous.dimshuffle(0, 1, 'x')
            obs = obs.dimshuffle(0, 1, 'x')
            return self.log_sum_exp(
                previous + obs + transition_matrix.dimshuffle('x', 0, 1),
                axis=1
            ) * mask.dimshuffle(0, 'x')

        initial = T.zeros_like(observations[0])
        beta, _ = theano.scan(
            fn=recurrence,
            outputs_info=[initial],
            sequences=[observations, mask],
            non_sequences=[transition_matrix]
        )

        return beta

    def _forward_backward(
        self,
        observations,
        mask
    ):
        """Call forward and backward to compute alpha and beta tables."""
        alpha = self._forward(observations, mask, self.transition_matrix)
        beta = self._backward(
            observations[::-1],
            mask[::-1],
            self.transition_matrix.T
        )

        return alpha, beta

    def _softmax_3d(self, x):
        e = T.exp(x - T.max(x, axis=-1, keepdims=True))
        s = T.sum(e, axis=-1, keepdims=True)
        return e / s

    def likelihood(
        self,
        observations,
        mask
    ):
        """Get CRF likelihoods from alpha and beta."""
        self.alpha, self.beta = self._forward_backward(observations, mask)
        self.gamma = self.alpha + self.beta + 1e-7
        return self._softmax_3d(self.gamma.dimshuffle(1, 0, 2))

    def batch_viterbi(
        self,
        observations,
    ):
        """Viterbi decoding with batch size > 1."""
        def recurrence(obs, previous, transition_matrix):
            previous = previous.dimshuffle(0, 1, 'x')
            obs = obs.dimshuffle(0, 'x', 1)
            scores = previous + obs + \
                transition_matrix.dimshuffle('x', 0, 1)
            return scores.max(axis=1), scores.argmax(axis=1)

        def sequence_function(beta_i, previous):
            return beta_i[T.arange(previous.shape[0]), previous]

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=(initial, None),
            sequences=[observations[1:]],
            non_sequences=[self.transition_matrix]
        )
        sequence, _ = theano.scan(
            fn=sequence_function,
            outputs_info=T.cast(T.argmax(self.alpha[0][-1], axis=1), 'int32'),
            sequences=T.cast(self.alpha[1][::-1], 'int32')
        )
        sequence = T.concatenate([
            sequence[::-1],
            [T.argmax(self.alpha[0][-1], axis=1)]
        ])
        return sequence

    def viterbi(
        self,
        observations
    ):
        """Viterbi decoding with batch-size 1."""
        def recurrence(obs, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            scores = previous + obs + self.transition_matrix
            return scores.max(axis=0), scores.argmax(axis=0)

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=(initial, None),
            sequences=[observations[1:]],
        )

        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32'),
            sequences=T.cast(self.alpha[1][::-1], 'int32')
        )

        sequence = T.concatenate([
            sequence[::-1],
            [T.argmax(self.alpha[0][-1])]
        ])

        return sequence

    def fprop(
        self,
        input,
        mask
    ):
        """Propogate input through the CRF."""
        input = input.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(1, 0)
        return self.likelihood(input, mask)
