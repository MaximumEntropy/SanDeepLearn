"""Implementations for different optimization techniques."""

import theano
import theano.tensor as T
import numpy as np

__author__ = "Sandeep Subramanian"
__maintainer__ = "Sandeep Subramanian"
__email__ = "sandeep.subramanian@gmail.com"


class Optimizer:
    """Optimization methods for backpropagation."""

    def __init__(self, scheduler=None, clip=5.0):
        """Initialize Optimizer with gradient clipping norm."""
        self.clip = clip
        self.scheduler = scheduler
        self.lr = theano.shared(np.float32(0.01).astype(theano.config.floatX)) \
            if self.scheduler is None else scheduler

    def sgd(self, cost, params, lr=0.01):
        """Stochatic Gradient Descent."""
        if self.scheduler is None:
            self.lr = theano.shared(np.float32(lr).astype(theano.config.floatX))

        gradients = T.grad(
            theano.gradient.grad_clip(cost, -1 * self.clip, self.clip),
            params
        )

        updates = []
        for param, gradient in zip(params, gradients):
            updates.append((param, param - self.lr * gradient))

        return updates

    def sgdmomentum(self, cost, params, lr=0.01, momentum=0.9):
        """Stochatic gradient descent with momentum."""
        assert 0 <= momentum < 1
        if self.scheduler is None:
            self.lr = theano.shared(np.float32(lr).astype(theano.config.floatX))
        momentum = theano.shared(
            np.float32(momentum).astype(theano.config.floatX)
        )

        gradients = T.grad(
            theano.gradient.grad_clip(cost, -1 * self.clip, self.clip),
            params
        )
        velocities = [theano.shared(np.zeros_like(
            param.get_value(borrow=True)
        ).astype(theano.config.floatX)) for param in params]

        updates = []

        for param, gradient, velocity in zip(params, gradients, velocities):
            new_velocity = momentum * velocity - self.lr * gradient
            updates.append((velocity, new_velocity))
            updates.append((param, param + new_velocity))

        return updates

    def adagrad(self, cost, params, lr=0.01, epsilon=1e-6):
        """Adaptive Gradient Optimization."""
        if self.scheduler is None:
            self.lr = theano.shared(np.float32(lr).astype(theano.config.floatX))
        epsilon = theano.shared(
            np.float32(epsilon).astype(theano.config.floatX)
        )

        gradients = T.grad(
            theano.gradient.grad_clip(cost, -1 * self.clip, self.clip),
            params
        )

        updates = []
        for param, gradient in zip(params, gradients):
            accumulated_gradient = theano.shared(
                np.zeros_like(
                    param.get_value(borrow=True)
                ).astype(np.float32),
                borrow=True
            )
            accumulated_gradient_new = accumulated_gradient + gradient ** 2
            updates.append((accumulated_gradient, accumulated_gradient_new))
            updates.append(
                (
                    param,
                    param - self.lr * gradient / T.sqrt(
                        accumulated_gradient_new + epsilon
                    )
                )
            )
        return updates

    def rmsprop(self, cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
        """RMSProp."""
        # http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        if self.scheduler is None:
            self.lr = theano.shared(np.float32(lr).astype(theano.config.floatX))
        epsilon = theano.shared(
            np.float32(epsilon).astype(theano.config.floatX)
        )
        rho = theano.shared(np.float32(rho).astype(theano.config.floatX))

        gradients = T.grad(
            theano.gradient.grad_clip(cost, -1 * self.clip, self.clip),
            params
        )

        updates = []
        for param, gradient in zip(params, gradients):
            accumulated_gradient = theano.shared(
                np.zeros_like(
                    param.get_value(borrow=True)
                ).astype(np.float32),
                borrow=True
            )
            accumulated_gradient_new = accumulated_gradient * rho + \
                gradient ** 2 * (1 - rho)
            updates.append((accumulated_gradient, accumulated_gradient_new))
            updates.append(
                (
                    param,
                    param - self.lr * gradient / T.sqrt(
                        accumulated_gradient_new + epsilon
                        )
                )
            )
        return updates

    def adam(
        self,
        cost,
        params,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    ):
        """ADAM."""
        # Reference - http://arxiv.org/pdf/1412.6980v8.pdf - Page 2
        if self.scheduler is None:
            self.lr = theano.shared(np.float32(lr).astype(theano.config.floatX))
        epsilon = theano.shared(
            np.float32(epsilon).astype(theano.config.floatX)
        )
        beta_1 = theano.shared(np.float32(beta_1).astype(theano.config.floatX))
        beta_2 = theano.shared(np.float32(beta_2).astype(theano.config.floatX))
        t = theano.shared(np.float32(1.0).astype(theano.config.floatX))

        gradients = T.grad(
            theano.gradient.grad_clip(cost, -1 * self.clip, self.clip),
            params
        )

        updates = []
        for param, gradient in zip(params, gradients):
            param_value = param.get_value(borrow=True)
            m_tm_1 = theano.shared(
                np.zeros_like(param_value).astype(np.float32),
                borrow=True
            )
            v_tm_1 = theano.shared(
                np.zeros_like(param_value).astype(np.float32),
                borrow=True
            )

            m_t = beta_1 * m_tm_1 + (1 - beta_1) * gradient
            v_t = beta_2 * v_tm_1 + (1 - beta_2) * gradient ** 2

            m_hat = m_t / (1 - beta_1)
            v_hat = v_t / (1 - beta_2)

            updated_param = param - (self.lr * m_hat) / (T.sqrt(v_hat) + epsilon)
            updates.append((m_tm_1, m_t))
            updates.append((v_tm_1, v_t))
            updates.append((param, updated_param))

        updates.append((t, t + 1.0))
        return updates
