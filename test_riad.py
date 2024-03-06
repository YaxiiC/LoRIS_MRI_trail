import os
import argparse
import matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pandas as pd
from tqdm import tqdm
import torch
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from utils.gen_mask import gen_mask
from utils.funcs import specificity_sensitivity, get_thresholds, save_res
from utils.plots import plot_scores_density, save_recon, plot_roc_curve, save_image, calculate_metrics
from models.tf_unet import load_trained_model
from datasets.tf_dataset import CustomDataGen, denormalize, normalize
from losses.tf_criterions import which_loss, COMBINED_W, Anomaly_Map, directional_difference
from utils.utils import AverageMeter
from sklearn.metrics import classification_report, confusion_matrix

@tf.function
def test_step(model, inputs):
    outputs = [model(x) for x in inputs]
    return outputs

def test(args, model, test_gen):
    scores = []
    anomaly_maps = []
    gt_mask_list = []
    reco_maps = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    mean_maps = []
    names_list = []
    for (data, label, box, names, mask) in tqdm(test_gen):
        mean_scores = np.array([0.] * args.batch_size)
        test_imgs.extend(data.numpy())
        gt_list.extend(label)
        names_list.extend(names)
        mean_scores = np.array([0.] * args.batch_size)
        score = 0
        for k in args.k_values:
            N = args.img_size // k
            Ms_generators = []
            Mss = []
            Ms_generator = gen_mask([k], 3, args.img_size)
            Ms = next(Ms_generator)
            for i in range(args.batch_size):
                Mss.append(Ms)
            data = np.asarray(data)
            inputs = []
            for j in range(3):
                temp_arr = []
                for i in range(len(data)):
                    m = Mss[i][j]
                    img = data[i]
                    temp_arr.append(((img+1)*m)-1)
                inputs.append(np.array(temp_arr))
            outputs = test_step(model, inputs)
            den_out = [o+tf.convert_to_tensor(1, dtype=tf.float32) for o in outputs]
            tot_mss = [1-sum([1-m for m in Ms]) for Ms in Mss]
            output_borders = sum(den_out)*tot_mss/3
            output_rects = []
            tot_mss_in = [sum([1-m for m in Ms]) for Ms in Mss]
            for j in range(len(Mss)):
                temp_rect = []
                for i in range(len(den_out)):
                    temp_rect.append(den_out[i][j]*tf.convert_to_tensor(1-Mss[j][i], dtype=tf.float32))
                output_rects.append(sum(temp_rect))

            output = []
            for i in range(len(output_borders)):
                    output.append((output_borders[i] + output_rects[i])/1.-tf.convert_to_tensor(1, dtype=tf.float32))
            if args.metric == 'msgms': 
                maps = Anomaly_Map(data, output)
                anomaly_scores = [mapp.max() for mapp in maps]
            else:
                maps = [directional_difference(d, o) for d, o in zip (data, output)]
                anomaly_scores = [m[co[1]:co[3], co[0]:co[2]].mean() for m, co in zip (maps, box)]
            mean_scores += np.array(anomaly_scores)
                
        reco_maps.extend(maps)
        anomaly_maps.extend(maps)
        
        for i, m in enumerate(mask):
            mask_array = m.numpy()
            if mask_array.shape[0] == 1:
                mask_array = mask_array.squeeze(0)
            if mask_array.ndim == 3:
                mask_array = mask_array[0]
            mask_array = (mask_array * 255).astype('uint8')

            anomaly_map_shape = anomaly_maps[i].shape
            num_channels = anomaly_map_shape[-1]

            mask_array_3d = np.repeat(mask_array[:, :, np.newaxis], num_channels, axis=2)
            
            gt_mask_list.append(mask_array_3d)
 
        scores.extend(mean_scores/len(args.k_values))
        recon_imgs.extend(output)
        
    return scores, test_imgs, recon_imgs, gt_list, reco_maps, names_list, gt_mask_list, anomaly_maps

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='RIAD anomaly detection')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--name', default='name')
    parser.add_argument('--k_values', type=int, nargs='+', default=[2, 4, 8, 16])
    parser.add_argument('--cv_fold', default=None)
    parser.add_argument('--phase', default='test')
    parser.add_argument('--metric', default='msgms')
    args = parser.parse_args()

    if not os.path.isdir('./runs/{}/{}_{}'.format(args.name, args.phase, args.metric)):
        os.mkdir('./runs/{}/{}_{}'.format(args.name, args.phase, args.metric))
        
    # -- set --
    loss = 'COMBINED'
    lam = 0.75

    # Load model
    loss_fn   = COMBINED_W(lam) if loss=='COMBINED_W' else which_loss(loss) 
    model     = load_trained_model(loss, loss_fn, args.name, 'best_model')
    
    # Test dataset
    data_path = '../datasets/anomaly' if args.cv_fold is None else '../datasets/anomaly_{}'.format(args.cv_fold)
    test_gen = CustomDataGen(data_path, class_name='sqr', is_train=False, phase=args.phase, resize=args.img_size, batch_size=args.batch_size)
    
    epoch_time = AverageMeter()
    start_time = time.time()

    # Start test
    scores, test_imgs, recon_imgs, gt_list, maps, names, gt_mask_list, anomaly_maps = test(args, model, test_gen)
    scores = np.asarray(scores)
    save_res(names, gt_list, scores, args.name, args.phase, args.metric)
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
    gt_list = np.asarray(gt_list)
    
    gt_mask_list = np.asarray(gt_mask_list)
    anomaly_maps =  np.asarray(anomaly_maps)

    plot_roc_curve(gt_list, scores, args.name, '{}_{}'.format(args.phase, args.metric))
    calculate_metrics(gt_mask_list, anomaly_maps)
    plot_scores_density(scores, ['max'], gt_list, args.name, '{}_{}'.format(args.phase, args.metric))
    if args.phase == 'test' and args.metric == 'dd':
        for index, (i, ir, m, s, n) in enumerate(zip(test_imgs, recon_imgs, maps, scores, names)):
            save_recon(i, ir, m, s, args.name, n)
    get_thresholds(gt_list, scores, args.name, args.phase, args.metric)