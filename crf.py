import theano
import theano.tensor as T

from utils import create_shared
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
