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
        Stochatic gradient descent.
        """
        lr = theano.shared(np.float32(lr).astype(theano.config.floatX))

        gradients = T.grad(cost, params)

        updates = []
        for param, gradient in zip(params, gradients):
            updates.append((param, param - lr * gradient))

        return updates