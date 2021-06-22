import tflib as lib

import numpy as np
import tensorflow as tf
tf.compat.v1.random.set_random_seed(1234)

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None


def Embedding(
    name,
    input_dim,
    output_dim,
    indices,
    initialization='glorot'
    ):
    """
    Creates an embedding matrix of dimensions input_dim x output_dim upon first use.
    Looks up embedding vector for each input symbol
    :parameters:
        name: name of embedding matrix tensor variable
        input_dim: No. of input symbols
        output_dim: Embedding dimension
        indices: input symbols tensor
    """
    
    with tf.name_scope(name) as scope:

        def uniform(stdev, size):
            if _weights_stdev is not None:
                stdev = _weights_stdev
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        if initialization == 'lecun':# and input_dim != output_dim):
            # disabling orth. init for now because it's too slow
            weight_values = uniform(
                np.sqrt(1./input_dim),
                (input_dim, output_dim)
            )

        elif initialization == 'glorot' or (initialization == None):

            weight_values = uniform(
                np.sqrt(2./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif initialization == 'he':

            weight_values = uniform(
                np.sqrt(2./input_dim),
                (input_dim, output_dim)
            )

        elif initialization == 'glorot_he':

            weight_values = uniform(
                np.sqrt(4./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif initialization == 'orthogonal' or \
            (initialization == None and input_dim == output_dim):
            
            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                        "supported.")
                flat_shape = (shape[0], np.prod(shape[1:]))
                    # TODO: why normal and not uniform?
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype('float32')
            weight_values = sample((input_dim, output_dim))

        else:

            raise Exception('Invalid initialization!')
            
        emb = lib.param(
            name+'.g',
            weight_values)

        return tf.nn.embedding_lookup(emb,indices)