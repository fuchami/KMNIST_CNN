# coding:utf-8
"""
独自オプティマイザー
https://github.com/yukiB/keras/commit/22702ff67e9f92adc341010a735e5765b478d80f
"""
import keras
import keras.backend as K
if K.backend() == 'tensorflow':
    import tensorflow as tf

def clip_norm(g, c, n):
    if c > 0:
        g = K.switch(n >= c, g * c / n, g)
    return g

class Optimizer(object):
    """Abstract optimizer base class.
    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.
    All Keras optimizers support the following keyword arguments:
        clipnorm: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        clipvalue: float >= 0. Gradients will be clipped
            when their absolute value exceeds this value.
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    def get_updates(self, params, constraints, loss):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def set_weights(self, weights):
        """Sets the weights of the optimizer, from Numpy arrays.
        Should only be called after computing the gradients
        (otherwise the optimizer has no weights).
        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the optimizer (i.e. it should match the
                output of `get_weights`).
        # Raises
            ValueError: in case of incompatible weight shapes.
        """
        params = self.weights
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Optimizer weight shape ' +
                                 str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        K.batch_set_value(weight_value_tuples)

    def get_weights(self):
        """Returns the current value of the weights of the optimizer.
        # Returns
            A list of numpy arrays.
        """
        return K.batch_get_value(self.weights)

    def get_config(self):
        config = {}
        if hasattr(self, 'clipnorm'):
            config['clipnorm'] = self.clipnorm
        if hasattr(self, 'clipvalue'):
            config['clipvalue'] = self.clipvalue
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class RMSpropGraves(Optimizer):
    """RMSPropGraves optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).
    This optimizer is usually a good choice for recurrent
    neural networks.
    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.001, rho=0.9, momentum=0.9, epsilon=1e-8, decay=0.,
                 **kwargs):
        super(RMSpropGraves, self).__init__(**kwargs)
        self.lr = K.variable(lr)
        self.rho = K.variable(rho)
        self.epsilon = epsilon
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)
        self.initial_decay = decay
        self.iterations = K.variable(0.)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        gs = [K.zeros(shape) for shape in shapes]
        moms = [K.zeros(shape) for shape in shapes] 
        self.weights = accumulators
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append(K.update_add(self.iterations, 1))

        for p, grad, g, mom, a in zip(params, grads, gs, moms, accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(grad)
            self.updates.append(K.update(a, new_a))
            new_g = self.rho * g + (1 - self.rho) * grad
            self.updates.append(K.update(g, new_g))
            #new_p = p - lr * grad / K.sqrt(new_a - K.square(new_g) +  self.epsilon)
            new_mom = self.momentum * mom - lr * grad / K.sqrt(new_a - K.square(new_g) +  self.epsilon)
            new_p = p + new_mom
            #new_p = p - lr * grad / (K.sqrt(new_a) + self.epsilon)

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(RMSpropGraves, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))