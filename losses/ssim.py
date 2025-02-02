import tensorflow as tf
import numpy as np 
import tensorflow_addons as tfa
from losses.iqa import wmsgms_map, gms_map, wgms_map

def compute_ssim_tf(data, outputs): 
    
    kernel_size = 11
    sigma = 2.0
    
    # Cast the data and outputs to float32 
    data = tf.cast(data, dtype=tf.float32) 
    outputs = tf.cast(outputs, dtype=tf.float32)

    # Apply Gaussian blur using TensorFlow Addons
    data_blur = tfa.image.gaussian_filter2d(data, filter_shape=kernel_size, sigma=sigma)
    outputs_blur = tfa.image.gaussian_filter2d(outputs, filter_shape=kernel_size, sigma=sigma)

    # Compute total variation using TensorFlow
    data_edges = tf.image.sobel_edges(data_blur)
    outputs_edges = tf.image.sobel_edges(outputs_blur)
    data_tv = tf.math.reduce_sum(tf.math.abs(data_edges), axis=-1)
    outputs_tv = tf.math.reduce_sum(tf.math.abs(outputs_edges), axis=-1)

    # Compute SSIM using the formula
    data_range = 1
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * data_tv + C1) * (2 * outputs_tv + C2)) / \
            ((data_tv ** 2 + outputs_tv ** 2 + C1) * (data_tv + outputs_tv + C2) + 1e-8)  

    return ssim_map

""" Define SSIM Anomaly Map """
def ssim_map(I, I_r): 
    similarity_map = compute_ssim_tf(I, I_r) 
    similarity_map = tfa.image.mean_filter2d(similarity_map) 
    anomaly_map = 1 - similarity_map.numpy()
    return anomaly_map

""" Define WSSIM Anomaly Map """
def wssim_map(I, I_r): 
    similarity_map = compute_ssim_tf(I, I_r) 
    GMS_map = gms_map(I, I_r) 
    anomaly_scores = [mapp.max() for mapp in (GMS_map)]
    anomaly_scores = np.array(anomaly_scores, dtype=np.float32)
    anomaly_scores_expanded = np.expand_dims(anomaly_scores, axis=-1) 
    anomaly_scores_expanded = np.expand_dims(anomaly_scores_expanded, axis=-1) 
    anomaly_scores_expanded = np.expand_dims(anomaly_scores_expanded, axis=-1) # Add a new dimension for width
    anomaly_map = (similarity_map) * (anomaly_scores_expanded)
    #anomaly_map = np.log(1 + np.abs(anomaly_map))
    return anomaly_map.numpy()

def compute_ms_ssim_tf(data, outputs, num_scales=2):
    ssim_maps = []
    
    for scale in range(num_scales):
        ssim_map = compute_ssim_tf(data, outputs)
        ssim_maps.append(ssim_map)
        
        # Downsample images for the next scale using average pooling
        data = tf.nn.avg_pool2d(data, ksize=2, strides=2, padding='SAME')
        outputs = tf.nn.avg_pool2d(outputs, ksize=2, strides=2, padding='SAME')
        
    return ssim_maps

def mssim_map(I, I_r):
    ssim_maps = compute_ms_ssim_tf(I, I_r)

    # Resize ssim_maps to a common shape using bilinear interpolation
    target_shape = ssim_maps[0].shape[1:3]
    resized_ssim_maps = [tf.image.resize(ssim_map, target_shape, method=tf.image.ResizeMethod.BILINEAR) for ssim_map in ssim_maps]

    # Calculate the weights for combining ssim_maps
    num_scales = len(ssim_maps)
    weights = [0.0448, 0.2856]  # Adjust the weights 

    # Combine ssim_maps with weights
    ms_ssim_map = sum(weight * ssim_map for weight, ssim_map in zip(weights, resized_ssim_maps))
    ms_ssim_map = ms_ssim_map/4
    ms_ssim_map = tfa.image.mean_filter2d(ms_ssim_map) 
    anomaly_map = 1 - ms_ssim_map.numpy()
    return anomaly_map

def wmssim_map(I, I_r):
    ms_ssim_map = mssim_map(I, I_r)
    #GMS_map = gms_map(I, I_r)
    similarity_map = ssim_map(I, I_r) 
    
    anomaly_scores = [mapp.max() for mapp in ms_ssim_map]
    anomaly_scores = np.array(anomaly_scores, dtype=np.float32)
    anomaly_scores_expanded = np.expand_dims(anomaly_scores, axis=-1) 
    anomaly_scores_expanded = np.expand_dims(anomaly_scores_expanded, axis=-1) 
    anomaly_scores_expanded = np.expand_dims(anomaly_scores_expanded, axis=-1) # Add a new dimension for width
 
    anomaly_map = ( similarity_map) * (anomaly_scores_expanded)
    anomaly_map = np.log(1 + np.abs(anomaly_map))
    
    return anomaly_map