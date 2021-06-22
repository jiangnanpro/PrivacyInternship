import tensorflow as tf
tf.compat.v1.random.set_random_seed(1234)

from tflib.ops.linear import Linear
from tflib.ops.conv2d import Conv2D
from tflib.ops.batchnorm import Batchnorm
from tflib.ops.deconv2d import Deconv2D
from tflib.ops.embedding import Embedding


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, DIM=64, OUTPUT_DIM=28*28, MODE='wgan-gp', noise=None):
    if noise is None:
        noise = tf.random.normal([n_samples, 128])

    output = Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = Batchnorm('Generator.BN3', [0,2,3], output)

    output = Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, INPUT_WIDTH=28, INPUT_HEIGHT=28, DIM=64, MODE='wgan-gp'):
    output = tf.reshape(inputs, [-1, 1, INPUT_WIDTH, INPUT_HEIGHT])

    output = Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

def ConditionalGenerator(n_samples, labels, embedding_dim=100, DIM=64, OUTPUT_DIM=28*28, MODE='wgan-gp', noise=None):
    assert labels.shape[0]==n_samples
    if noise is None:
        noise = tf.random.normal([n_samples, 128])
    else:
        assert labels.shape[0]==noise.shape[0]
    label_embedding = Embedding('ConditionalGenerator.Embedding', 10, embedding_dim, labels)
    
    noise_labels = tf.concat([noise, label_embedding],1)

    #embeddings = tf.keras.layers.Embedding(10, 10)
    #embed = embeddings(words_ids)

    output = Linear('ConditionalGenerator.Input', 128+embedding_dim, 4*4*4*DIM, noise_labels)
    if MODE == 'wgan':
        output = Batchnorm('ConditionalGenerator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = Deconv2D('ConditionalGenerator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = Batchnorm('ConditionalGenerator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = Deconv2D('ConditionalGenerator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = Batchnorm('ConditionalGenerator.BN3', [0,2,3], output)

    output = Deconv2D('ConditionalGenerator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def ConditionalDiscriminator(inputs, labels, embedding_dim=100, INPUT_WIDTH=28, INPUT_HEIGHT=28, DIM=64, MODE='wgan-gp'):
    assert labels.shape[0]==inputs.shape[0]
    labels_in = Embedding('ConditionalDiscriminator.Embedding', 10, embedding_dim, labels)
    labels_in = Linear('ConditionalDiscriminator.Labels', embedding_dim, INPUT_WIDTH*INPUT_HEIGHT, labels_in)
    labels_in = tf.reshape(labels_in, [-1, 1, INPUT_WIDTH, INPUT_HEIGHT])

    images_in = tf.reshape(inputs, [-1, 1, INPUT_WIDTH, INPUT_HEIGHT])

    output = tf.concat([images_in, labels_in], axis=1)


    output = Conv2D('ConditionalDiscriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = Conv2D('ConditionalDiscriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = Batchnorm('ConditionalDiscriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = Conv2D('ConditionalDiscriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = Batchnorm('ConditionalDiscriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = Linear('ConditionalDiscriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

Variables found in checkpoint file [('ConditionalDiscriminator.1/ConditionalDiscriminator.1.Biases', [64]), 
('ConditionalGenerator.2/ConditionalGenerator.2.Biases', [128]), 
('ConditionalGenerator.2/ConditionalGenerator.2.Biases/Adam', [128]), 
('ConditionalGenerator.2/ConditionalGenerator.2.Biases/Adam_1', [128]), 
('ConditionalGenerator.2/ConditionalGenerator.2.Filters', [5, 5, 128, 256]), 
('ConditionalGenerator.2/ConditionalGenerator.2.Filters/Adam', [5, 5, 128, 256]), 
('ConditionalGenerator.2/ConditionalGenerator.2.Filters/Adam_1', [5, 5, 128, 256]), 
('ConditionalGenerator.3/ConditionalGenerator.3.Biases', [64]), 
('ConditionalGenerator.3/ConditionalGenerator.3.Biases/Adam', [64]), 
('ConditionalGenerator.3/ConditionalGenerator.3.Biases/Adam_1', [64]), 
('ConditionalGenerator.3/ConditionalGenerator.3.Filters', [5, 5, 64, 128]), 
('ConditionalGenerator.3/ConditionalGenerator.3.Filters/Adam', [5, 5, 64, 128]), 
('ConditionalGenerator.3/ConditionalGenerator.3.Filters/Adam_1', [5, 5, 64, 128]), 
('ConditionalGenerator.5/ConditionalGenerator.5.Biases', [1]), 
('ConditionalGenerator.5/ConditionalGenerator.5.Biases/Adam', [1]), 
('ConditionalGenerator.5/ConditionalGenerator.5.Biases/Adam_1', [1]), 
('ConditionalGenerator.5/ConditionalGenerator.5.Filters', [5, 5, 1, 64]), 
('ConditionalGenerator.5/ConditionalGenerator.5.Filters/Adam', [5, 5, 1, 64]), 
('ConditionalGenerator.5/ConditionalGenerator.5.Filters/Adam_1', [5, 5, 1, 64]), 
('ConditionalGenerator.Embedding/ConditionalGenerator.Embedding.g', [10, 10]), 
('ConditionalGenerator.Embedding/ConditionalGenerator.Embedding.g/Adam', [10, 10]), 
('ConditionalGenerator.Embedding/ConditionalGenerator.Embedding.g/Adam_1', [10, 10]), 
('ConditionalGenerator.Input/ConditionalGenerator.Input.W', [138, 4096]), 
('ConditionalGenerator.Input/ConditionalGenerator.Input.W/Adam', [138, 4096]), 
('ConditionalGenerator.Input/ConditionalGenerator.Input.W/Adam_1', [138, 4096]), 
('ConditionalGenerator.Input/ConditionalGenerator.Input.b', [4096]), 
('ConditionalGenerator.Input/ConditionalGenerator.Input.b/Adam', [4096]), 
('ConditionalGenerator.Input/ConditionalGenerator.Input.b/Adam_1', [4096]), 
('beta1_power', []), ('beta1_power_1', []), ('beta2_power', []), ('beta2_power_1', [])]