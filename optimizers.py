#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy as np

__author__ = "Sandeep Subramanian"
__maintainer__ = "Sandeep Subramanian"
__email__ = "sandeep.subramanian@gmail.com"

class Optimizer:
    
    """
    Optimization methods for backpropagation
    """

    def __init__(self):

        """
        __TODO__ add gradient clipping
        """

    def sgd(self, cost, params, lr=0.01):
        """
        Stochatic Gradient Descent.
        """
        lr = theano.shared(np.float32(lr).astype(theano.config.floatX))

        gradients = T.grad(cost, params)

        updates = []
        for param, gradient in zip(params, gradients):
            updates.append((param, param - lr * gradient))

        return updates

    def adagrad(self, cost, params, lr=0.01, epsilon=1e-6):
        """
        Adaptive Gradient Optimization.
        """
        lr = theano.shared(np.float32(lr).astype(theano.config.floatX))
        epsilon = theano.shared(np.float32(epsilon).astype(theano.config.floatX))

        gradients = T.grad(cost, params)

        updates = []
        for param, gradient in zip(params, gradients):
            accumulated_gradient = theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(np.float32), borrow=True)
            accumulated_gradient_new = accumulated_gradient + gradient  ** 2
            updates.append((accumulated_gradient, accumulated_gradient_new))
            updates.append((param, param - lr * gradient / T.sqrt(accumulated_gradient_new + epsilon)))
        return updates

    def rmsprop(self, cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
        """
        RMSProp
        Reference - http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        """

        lr = theano.shared(np.float32(lr).astype(theano.config.floatX))
        epsilon = theano.shared(np.float32(epsilon).astype(theano.config.floatX))
        rho = theano.shared(np.float32(rho).astype(theano.config.floatX))

        gradients = T.grad(cost, params)

        updates = []
        for param, gradient in zip(params, gradients):
            accumulated_gradient = theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(np.float32), borrow=True)
            accumulated_gradient_new = accumulated_gradient * rho + gradient ** 2 * (1 - rho)
            updates.append((accumulated_gradient, accumulated_gradient_new))
            updates.append((param, param - lr * gradient / T.sqrt(accumulated_gradient_new + epsilon)))
        return updates
