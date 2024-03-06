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

def directional_difference(image1, image2):
    if image1.shape[2] == 3:
        image1 = color.rgb2gray(denormalize(image1))
    if image2.shape[2] == 3:
        image2 = color.rgb2gray(denormalize(image2))
    diff = np.maximum(image2 - image1, 0)
    return diff

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

# get right loss function
def which_loss(loss_fn):
    switcher = {
        'mse':      "mse",
        'L2':       L2_Loss,
        'SSIM':     SSIM_Loss,
        'MSGMS':    MSGMS_Loss,
        'COMBINED': COMBINED_Loss,
    }
    return switcher.get(loss_fn, "Invalid loss function")


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
    g_I   = Grad_Mag_Map(I)
    g_Ir  = Grad_Mag_Map(I_r)
    similarity_map = (2 * g_I * g_Ir + c) / (g_I**2 + g_Ir**2 + c)
    if show:
        similarity_map = tf.squeeze(similarity_map, axis=0).numpy()
    return similarity_map


""" Gradient Magnitude Distance Map"""
def GMS_Loss(I, I_r):
    x = tf.reduce_mean(1 - GMS(I, I_r)) 
    return x


#### LOSS FUNCTIONS ####
""" Define MSGMS """
def MSGMS_Loss(I, I_r):
    # normal scale loss
    tot_loss = GMS_Loss(I, I_r)
    # pool 3 times and compute GMS
    for _ in range(3):
        I   = tf.nn.avg_pool2d(I,   ksize=2, strides=2, padding= 'VALID')
        I_r = tf.nn.avg_pool2d(I_r, ksize=2, strides=2, padding= 'VALID')
        # sum loss
        tot_loss += GMS_Loss(I, I_r)

    return tot_loss/4


""" Define SSIM loss"""
def SSIM_Loss(I, I_r):
    I   = tf.cast(I,   dtype=tf.double)
    I_r = tf.cast(I_r, dtype=tf.double)
    img_range = 1+1#tf.reduce_max(1)-tf.reduce_min(X_train)
    ssim = tf.image.ssim(I, I_r, max_val=img_range, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return tf.reduce_mean(1 - ssim)

def ssim_Loss(I, I_r):
    # Compute the SSIM between the two images
    ssim = tf.image.ssim(I, I_r, max_val=1.0)
    
    # Calculate the mean SSIM value over all pixels
    loss = 1 - tf.reduce_mean(ssim)
    
    return loss

""" Define l2 loss"""
def L2_Loss(I, I_r):
    l2_loss = tf.keras.losses.MeanSquaredError()
    return l2_loss(I, I_r)

def l2_loss(I, I_r):
    return tf.math.reduce_mean(tf.math.squared_difference(I, I_r))

""" Define total loss"""  
def COMBINED_Loss(I, I_r, lambda_s=1.0, lambda_m=1.0):
    l2_loss = L2_Loss(I, I_r)
    S_loss  = SSIM_Loss(I, I_r)
    M_loss  = MSGMS_Loss(I, I_r)

    x = l2_loss + lambda_s * S_loss + lambda_m * M_loss 
    return [tf.reduce_mean(x), tf.reduce_mean(M_loss), tf.reduce_mean(S_loss), tf.reduce_mean(l2_loss)]

def COMBINED_W(lambda_s):
    def COMBINED_W_Loss(I, I_r):
        l2_loss = l2_loss(I, I_r)
        S_loss  = ssim_Loss(I, I_r)
        M_loss  = MSGMS_Loss(I, I_r)
        
        lambda_m = 1 - lambda_s

        x = lambda_s*S_loss + lambda_m*M_loss + l2_loss
        return tf.reduce_mean(x)
    return COMBINED_Loss

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
        # upsample
        gms_scale = tf.image.resize(gms_scale, size=size)
        gms_tot  += gms_scale

    gms_map = gms_tot/4
    gms_map = tfa.image.mean_filter2d(gms_map) 
    return gms_map



""" Define MSGMS Anomaly Map """
def Anomaly_Map(I, I_r):
    return (1 - MSGMS_Map(I, I_r)).numpy()