import time
import random
import os
import yaml
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import cv2
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
#from osgeo import gdal


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def generate_run(name):
    runs_path = './runs'
    run_path = runs_path+'/{}'.format(name)
    figures_path =  run_path+'/figures'
    train_fig_path = run_path+'/train_figures'
    models_path =  run_path+'/models'
    reconstructed_path =  run_path+'/reconstructed'
    if not os.path.isdir(runs_path):
        os.mkdir(runs_path)
    if not os.path.isdir(run_path):
        os.mkdir(run_path)
        os.mkdir(figures_path)
        os.mkdir(train_fig_path)
        os.mkdir(models_path)
        os.mkdir(reconstructed_path)
    print('Run saved in {}'.format(run_path))
    return run_path

def save_args(args, path):
    args_dict = vars(args)

    with open('{}/args.yaml'.format(path), 'w') as f:
        yaml.dump(args_dict, f)

def crf(original_image, annotated_image, use_2d = True):
    
    original_image = np.uint8(255 * original_image)
    if(len(annotated_image.shape)<3):
        annotated_image = gray2rgb(np.uint8(annotated_image*255)).astype(np.uint32)
    
    annotated_image = annotated_image.astype(np.uint32)
    annotated_label = annotated_image[:,:,0].astype(np.uint32) + (annotated_image[:,:,1]<<8).astype(np.uint32) + (annotated_image[:,:,2]<<16).astype(np.uint32)
    
    colors, labels = np.unique(annotated_label, return_inverse=True)
    
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    n_labels = len(set(labels.flat)) 
    
    print("No of labels in the Image are ")
    print(n_labels)
    
    
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        U = unary_from_labels(labels, n_labels, gt_prob=0.90, zero_unsure=False)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)

    MAP = np.argmax(Q, axis=0)

    MAP = colorize[MAP,:]
    return MAP.reshape(original_image.shape)
