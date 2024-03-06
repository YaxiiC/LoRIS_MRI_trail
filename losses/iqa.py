import tensorflow as tf
import tensorflow_addons as tfa
from skimage import io, color
import sys
from datasets.tf_dataset_ import denormalize
import numpy as np

IMG_WIDTH    = 256
IMG_HEIGHT   = 256
IMG_CHANNELS = 3
size = (IMG_WIDTH,IMG_HEIGHT)

# modify source of prewitt
def pad(input, ksize, mode, constant_values):
    input = tf.convert_to_tensor(input)
    ksize = tf.convert_to_tensor(ksize)
    mode = "CONSTANT" if mode is None else upper(mode)
    constant_values = (
        tf.zeros([], dtype=input.dtype)
        if constant_values is None
        else tf.convert_to_tensor(constant_values, dtype=input.dtype)
    )

    assert mode in ("CONSTANT", "REFLECT", "SYMMETRIC")

    height, width = ksize[0], ksize[1]
    top = (height - 1) // 2
    bottom = height - 1 - top
    left = (width - 1) // 2
    right = width - 1 - left
    paddings = [[0, 0], [top, bottom], [left, right], [0, 0]]
    return tf.pad(input, paddings, mode=mode, constant_values=constant_values)


def prewitt(input, mode=None, constant_values=None, name=None):

    input = tf.convert_to_tensor(input)

    gx = tf.cast([[1, 0, -1], [1, 0, -1], [1, 0, -1]], input.dtype)
    gy = tf.cast([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], input.dtype)

    ksize = tf.constant([3, 3])

    input = pad(input, ksize, mode, constant_values)

    channel = tf.shape(input)[-1]
    shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
    gx, gy = tf.reshape(gx, shape), tf.reshape(gy, shape)
    shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
    gx, gy = tf.broadcast_to(gx, shape), tf.broadcast_to(gy, shape)

    x = tf.nn.depthwise_conv2d(
        input, tf.cast(gx, input.dtype), [1, 1, 1, 1], padding="VALID"
    )
    y = tf.nn.depthwise_conv2d(
        input, tf.cast(gy, input.dtype), [1, 1, 1, 1], padding="VALID"
    )
    epsilon = 1e-08
    return tf.math.sqrt(x * x + y * y + sys.float_info.epsilon) 

""" Gradient Magnitude Map """
def Grad_Mag_Map(I, show = False):
    I = tf.reduce_mean(I, axis=-1, keepdims=True)
    I = tfa.image.median_filter2d(I, filter_shape=(3, 3), padding='REFLECT')
    x = prewitt(I)
    if show:
        x = tf.squeeze(x, axis=0).numpy()
    return x


""" Gradient Magnitude Similarity Map"""
def GMS(I, I_r, show=False, c=0.0026):
    # Convert inputs to grayscale
    I   = tf.image.rgb_to_grayscale(I)
    I_r = tf.image.rgb_to_grayscale(I_r)
 
    g_I   = Grad_Mag_Map(I)
    g_Ir  = Grad_Mag_Map(I_r)
    similarity_map = (2 * g_I * g_Ir + c) / (g_I**2 + g_Ir**2 + c)
    if show:
        similarity_map = tf.squeeze(similarity_map, axis=0).numpy()
    return similarity_map

""" Define GMS Map """
def gms_map(I, I_r):
    I = tf.cast(I, dtype=tf.float32)
    I_r = tf.cast(I_r, dtype=tf.float32)
    
    # normal scale similarity map
    gms_map = GMS(I, I_r)

    gms_map = tfa.image.mean_filter2d(gms_map) 
    anomaly_map =  1 - gms_map.numpy()
    return anomaly_map

""" Define SSIM GMS Map """
def wgms_map(I, I_r):
    I = tf.cast(I, dtype=tf.float32)
    I_r = tf.cast(I_r, dtype=tf.float32)
    I   = tf.nn.avg_pool2d(I, ksize=2, strides=2, padding= 'VALID')
    I_r = tf.nn.avg_pool2d(I_r, ksize=2, strides=2, padding= 'VALID')
    # normal scale similarity map
    gms_map = GMS(I, I_r)

    gms_map = tfa.image.mean_filter2d(gms_map) 
    gms_map = tf.image.resize(gms_map, size=size)
    # Compute the SSIM between the original and distorted images
    ssim_value = tf.image.ssim(I, I_r, max_val=1.0)
    
    # Convert SSIM value to a tensor
    ssim_tensor = tf.convert_to_tensor(ssim_value, dtype=tf.float32)
    
    #anomaly_map = (1 - gms_map) * (1 - ssim_tensor)
    anomaly_map = (1 - gms_map)  * tf.reshape((1 - ssim_tensor), (4, 1, 1, 1))
    anomaly_map = np.log(1 + np.abs(anomaly_map))
    
    return anomaly_map

""" Define MSGMS Map """
def MSGMS_Map(I, I_r):
    I   = tf.cast(I, dtype=tf.float32)
    I_r = tf.cast(I_r, dtype=tf.float32)
    # normal scale similarity map
    gms_tot = GMS(I, I_r)

    # pool 3 times and compute GMS
    for _ in range(3):
        I   = tf.nn.avg_pool2d(I, ksize=2, strides=2, padding= 'VALID')
        I_r = tf.nn.avg_pool2d(I_r, ksize=2, strides=2, padding= 'VALID')
        # compute GMS 
        gms_scale = GMS(I, I_r)
        gms_scale = tfa.image.mean_filter2d(gms_scale)
        # upsample
        gms_scale = tf.image.resize(gms_scale, size=size)
        gms_tot  += gms_scale

    gms_map = gms_tot/4
    gms_map = tfa.image.mean_filter2d(gms_map) 
    return gms_map



# Define MSGMS Anomaly Map with SSIM
def wmsgms_map(I, I_r):
    
    msgms_map = MSGMS_Map(I, I_r)
    msgms_map = tfa.image.mean_filter2d(msgms_map)
    # Compute the SSIM between the original and distorted images
    ssim_value = tf.image.ssim(I, I_r, max_val=1.0)
    
    # Convert SSIM value to a tensor
    ssim_tensor = tf.convert_to_tensor(ssim_value, dtype=tf.float32)
    
    anomaly_map = (1 - msgms_map)  + tf.reshape((1 - ssim_tensor), (4, 1, 1, 1))
    
    #anomaly_map = (1 - msgms_map) * (1 - ssim_tensor)
    
    #anomaly_map = np.log(1 + np.abs(anomaly_map))
    print('Anomaly_map: {}'.format(anomaly_map.shape))
    print('SSIM: {}'.format(ssim_tensor.shape))
    print('msgms_map: {}'.format(msgms_map.shape))
    return anomaly_map.numpy()
