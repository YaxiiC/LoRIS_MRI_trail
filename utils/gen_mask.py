import numpy as np
import torch
import tensorflow as tf
import cv2

def gen_mask(k_list, n, im_size):
    while True:
        Ms = []
        for k in k_list:
            N = im_size // k
            rdn = np.random.permutation(N**2)
            additive = N**2 % n
            if additive > 0:
                rdn = np.concatenate((rdn, np.asarray([-1] * (n - additive))))
            n_index = rdn.reshape(n, -1)
            for index in n_index:
                full_tmp = np.full((im_size, im_size, 3), 1, dtype=np.float32)
                tmp = [0 if i in index else 1 for i in range(N**2)]
                tmp = np.asarray(tmp).reshape(N, N)
                tmp = tmp.repeat(k, 0).repeat(k, 1)
                full_tmp[:,:,0] = tmp
                full_tmp[:,:,1] = tmp
                full_tmp[:,:,2] = tmp
                Ms.append(full_tmp)
        yield Ms

def gen_mask_boxed(k_list, n, im_size, box):
    #box = bb[0]
    while True:
        Ms = []
        for k in k_list:
            N1 = abs(box[3]-box[1]) // k
            N2 = abs(box[2]-box[0]) // k
            rdn = np.random.permutation(N1*N2)
            additive = N1*N2 % n
            if additive > 0:
                rdn = np.concatenate((rdn, np.asarray([-1] * (n - additive))))
            n_index = rdn.reshape(n, -1)
            for index in n_index:
                tmp = [0 if i in index else 1 for i in range(N1*N2)]
                tmp = np.asarray(tmp).reshape(N1, N2)
                tmp = tmp.repeat(k, 0).repeat(k, 1)
                x = box[0]
                y = box[1]
                white_image = np.full((im_size, im_size, 3), 1, dtype=np.float32)
                white_image[y:y+tmp.shape[0], x:x+tmp.shape[1], 0] = tmp
                white_image[y:y+tmp.shape[0], x:x+tmp.shape[1], 1] = tmp
                white_image[y:y+tmp.shape[0], x:x+tmp.shape[1], 2] = tmp
                Ms.append(white_image)
        yield Ms
        
def gen_mask_single(image, mask_coords = [110, 150, 150, 110]):
    mask_color = (0, 0, 0) # black
    x, y, w, z = mask_coords
    copy_image = image.copy()
    if tf.is_tensor(image):                                   # convert to numpy for masking
        copy_image = copy_image.numpy() + 1
    cv2.rectangle(copy_image, (x, y), (w, z), mask_color, -1) # mask
    if tf.is_tensor(image):                                   # back to tensor
        copy_image = tf.constant(copy_image)
    return copy_image