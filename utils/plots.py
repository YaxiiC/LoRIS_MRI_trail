import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import roc_curve
import numpy as np
import cv2
import tensorflow as tf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_labels



def denormalize(images):
    return (tf.round(tf.multiply(images + 1, 127.5))/255).numpy()

def plot_scores_density(scores, params, labels, name, phase):

    df = pd.DataFrame()
    df['scores'] = scores
    df['labels'] = labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    sns.kdeplot(data=df, x="scores", hue='labels', fill=True, common_norm=False, alpha=0.8, ax=ax1)
    ax1.set_xlabel("Anomaly score")
    ax1.set_title(params[0])
    plt.show()
    plt.savefig('./runs/{}/{}/density.eps'.format(name, phase))

def get_refined_mask(image, th):
    binary = image > th
    binary = (binary*255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    mask = (dilated/255.)
    return dilated, mask

def save_image(x, xm, xr, xrsum, epoch, num, path, name='train', labels=['Original', 'Masked', 'Reconstructed', 'Reconstructed_sum']):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,6))
    
    # plot masked
    ax1.imshow(denormalize(x), interpolation='nearest')
    if 'batch' in name:
        ax1.title.set_text('1')
    else:
        ax1.title.set_text(labels[0])
    # plot reconstructed
    ax2.imshow(denormalize(xm), interpolation='nearest')
    if 'batch' in name:
        ax2.title.set_text('2')
    else:
        ax2.title.set_text(labels[1])
    # plot reconstructed
    ax3.imshow(denormalize(xr), interpolation='nearest')
    if 'batch' in name:
        ax3.title.set_text('3')
    else:
        ax3.title.set_text(labels[2])

    ax4.imshow(denormalize(xrsum), interpolation='nearest')
    if 'batch' in name:
        ax4.title.set_text('4')
    else:
        ax4.title.set_text(labels[3])
    fig.savefig('{}/train_figures/{}_{}_{}.png'.format(path, name, epoch, num))
    
def save_recon(i, ir, m, s, name, index, th):
    dilated, mask = get_refined_mask(m, th)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,8))
    
    # plot masked
    ax1.imshow(denormalize(i), interpolation='nearest')
    ax1.title.set_text('Original')
    
    # plot original
    ax2.imshow(denormalize(ir), interpolation='nearest')
    ax2.title.set_text('Reconstructed')

    # plot reconstructed 1
    ax3.imshow(m, cmap='jet', interpolation='nearest')
    ax3.title.set_text('Map')

    ax4.imshow(dilated, interpolation='nearest', cmap='gray', vmin=0, vmax=1)
    ax4.title.set_text('Segmentation')

    plt.suptitle("Anomaly score: {}".format(s))
    fig.show()
    plt.savefig('./runs/{}/reconstructed/test_{}.png'.format(name, index))
    plt.close()

def plot_roc_curve(gt_list, scores, name, phase):

    fpr, tpr, trh = roc_curve(gt_list, scores)
    img_roc_auc = roc_auc_score(gt_list, scores)
    print('image ROCAUC: %.3f' % (img_roc_auc))
    plt.plot(fpr, tpr, label='Img_ROCAUC: %.3f' % (img_roc_auc))
    plt.legend(loc="lower right")
    plt.savefig('./runs/{}/{}/ROCAUC.eps'.format(name, phase))

def calculate_metrics(gt_mask_list, anomaly_maps):
    # Flatten and convert ground truth masks and anomaly maps
    flat_ground_truth_masks = gt_mask_list.ravel().astype(int)
    gt_mask_binary = np.where(flat_ground_truth_masks > 0, 1, 0)
    flat_anomaly_maps = anomaly_maps.ravel()
    
    # Calculate precision and recall
    precision, recall, thresholds = precision_recall_curve(gt_mask_binary, flat_anomaly_maps)
    # Calculate Dice 
    f1_scores = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)


    # Calculate Pixel ROC AUC
    per_pixel_rocauc = roc_auc_score(gt_mask_binary, flat_anomaly_maps) # flat_anomaly_maps
    print('Pixel ROCAUC: %.3f' % per_pixel_rocauc)

    dice_coefficient = f1_scores[np.argmax(f1_scores)]
    print('Dice: %.3f' % dice_coefficient)
    print('threshold: ', thresholds[np.argmax(f1_scores)])
    return thresholds[np.argmax(f1_scores)]

def calculate_metrics2(gt_mask_list, anomaly_maps, th):

    new_anomaly_maps = []
    for m in anomaly_maps:
       _, map_refined =  get_refined_mask(m, th)
       new_anomaly_maps.append(map_refined)
    new_anomaly_maps = np.array(new_anomaly_maps)

    calculate_metrics(gt_mask_list, new_anomaly_maps)

def get_seg(image, th):
    image = np.where(image > th, 255, 0)
    return image
