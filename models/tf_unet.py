import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations

''' pre-processing functions '''
IMG_WIDTH    = 256
IMG_HEIGHT   = 256
IMG_CHANNELS = 3
size = (IMG_WIDTH,IMG_HEIGHT)

# Building blocks
def EncoderBlock(inputs, out_channels, kernel_size, stride):
    ## miniblock 1
    c1 = layers.Conv2D(out_channels, kernel_size, stride, 'same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation(activations.relu)(c1)
    # miniblock 2
    c1 = layers.Conv2D(out_channels, 3, 1, 'same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation(activations.relu)(c1)
    return c1

def DecoderBlock(up_input, down_input, out_channels):
    up_input = layers.Conv2DTranspose(out_channels, 4, 2, 'same')(up_input)
    inputs = layers.concatenate([up_input, down_input])

    ## miniblock 1
    c1 = layers.Conv2D(out_channels, 3, 1, 'same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation(activations.relu)(c1)
    # miniblock 2
    c1 = layers.Conv2D(out_channels, 3, 1, 'same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation(activations.relu)(c1)
    return c1



# Model
def AutoEncoder(activation):
    
    # input layer with normalization and standardization
    inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    latent_dim = 512

    # Encoding path
    e1 = EncoderBlock(inputs,  64, 3, 1)
    e2 = EncoderBlock(e1,     128, 4, 2)
    e3 = EncoderBlock(e2,     256, 4, 2)

    # Latent dimension
    e4 = EncoderBlock(e3, latent_dim, 4, 2)
    e5 = EncoderBlock(e4, latent_dim, 4, 2)
    d1 = DecoderBlock(e5, e4, latent_dim)

    # Dencoding path 
    d2 = DecoderBlock(d1, e3, 256)
    d3 = DecoderBlock(d2, e2, 128)
    d4 = DecoderBlock(d3, e1, 64)

    # final convolution 
    d5 = layers.Conv2D(IMG_CHANNELS, 3, 1, 'same')(d4)

    if activation   == 'relu':
        outputs = tf.keras.activations.relu(d5)
    elif activation == 'tanh':
        outputs = layers.Activation(activations.tanh)(d5)
    elif activation == 'sigmoid': 
        outputs = tf.keras.activations.sigmoid(d5)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

def load_trained_model(loss, loss_fn, path, name):
    model   = tf.keras.models.load_model('./runs/{}/models/{}'.format(path, name), custom_objects={loss+'_Loss': loss_fn})
    return model