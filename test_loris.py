import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
from utils.gen_mask import gen_mask_single
from utils.funcs import get_thresholds, save_res
from utils.plots import calculate_metrics2, calculate_metrics, plot_scores_density, save_recon, plot_roc_curve, get_seg
from weights.tf_unet import load_trained_model
from datasets.tf_dataset import CustomDataGen
from losses.tf_criterions import *
from losses.iqa import wmsgms_map, gms_map, wgms_map
from losses.ssim import ssim_map, mssim_map, wssim_map, wmssim_map
from losses.iwssim import iwssim_map
from utils.utils import AverageMeter
from utils.utils import crf

@tf.function
def test_step(model, inputs):
    outputs = model(inputs)
    return outputs


def test(args, model, test_gen):
    scores = []
    anomaly_maps = []
    ref_anomaly_maps = []
    reco_maps = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    names_list = []
    
    print("Total number of batches:", len(test_gen))

    for (data, label, box, names, mask) in tqdm(test_gen):
        data_np = data.numpy()
        test_imgs.extend(data_np)
        names_list.extend(names)
        gt_list.extend(label)
 
        inputs = []
                  
        data = np.asarray(data)
        for i, b in zip(data, box):
            inputs.append(gen_mask_single(i+1, b))
        inputs = np.asarray(inputs)-1
        
        with tf.GradientTape() as tape:
            outputs = test_step(model, inputs) 
        if args.metric == 'msgms': 
            maps = Anomaly_Map(data, outputs)
            anomaly_scores = [mapp.max() for mapp in maps]
        elif args.metric == 'wmsgms': 
            maps = wmsgms_map(data, outputs)
            anomaly_scores = [mapp.max() for mapp in maps]
        elif args.metric == 'mssim': 
            maps = mssim_map(data, outputs)
            anomaly_scores = [mapp.max() for mapp in maps]
        elif args.metric == 'ssim': 
            maps = ssim_map(data, outputs)
            anomaly_scores = [mapp.max() for mapp in maps]
        elif args.metric == 'gms': 
            maps = gms_map(data, outputs)
            anomaly_scores = [mapp.max() for mapp in maps]
        elif args.metric == 'wgms': 
            maps = wgms_map(data, outputs)
            anomaly_scores = [mapp.max() for mapp in maps]
        elif args.metric == 'wmssim': 
            maps = wmssim_map(data, outputs)
            anomaly_scores = [mapp.max() for mapp in maps]
        elif args.metric == 'wssim': 
            maps = wssim_map(data, outputs)
            anomaly_scores = [mapp.max() for mapp in maps]
        else:
            maps = [directional_difference(d, o) for d, o in zip (data, outputs)]    
            anomaly_scores = [m[co[1]:co[3], co[0]:co[2]].max() for m, co in zip (maps, box)]
        
        reco_maps.extend(maps)
        scores.extend(anomaly_scores)
        anomaly_maps.extend(maps)
        
       
        for i, m in enumerate(mask):
            mask_array = m.numpy()
            if mask_array.shape[0] == 1:
                mask_array = mask_array.squeeze(0)
            if mask_array.ndim == 3:
                mask_array = mask_array[0]
            mask_array = (mask_array * 255).astype('uint8')
           
            gt_mask_list.append(mask_array)

        recon_imgs.extend(outputs)

    return scores, test_imgs, recon_imgs, gt_list, reco_maps, names_list, gt_mask_list, anomaly_maps





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='LoRIS weakly supervised anomaly detection')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--name', default='name')
    parser.add_argument('--cv_fold', default=None)
    parser.add_argument('--phase', default='test')
    parser.add_argument('--metric', default='msgms')
    args = parser.parse_args()

    if not os.path.isdir('./runs/{}/{}_{}'.format(args.name, args.phase, args.metric)):
        os.mkdir('./runs/{}/{}_{}'.format(args.name, args.phase, args.metric))
        
    loss = 'COMBINED'
    lam = 0.75

    # Load model
    loss_fn   = COMBINED_W(lam) if loss=='COMBINED_W' else which_loss(loss) 
    model     = load_trained_model(loss, loss_fn, args.name, 'best_model')

    # Test dataset
    data_path = '../anomaly' if args.cv_fold is None else '../anomaly_{}'.format(args.cv_fold)

    test_gen = CustomDataGen(data_path, class_name='sqr', is_train=False, phase=args.phase, resize=args.img_size, batch_size=args.batch_size, fold=args.cv_fold)

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
    test_imgs_list =  np.asarray(test_imgs)
    recon_imgs_list = np.asarray(recon_imgs)
    
    plot_roc_curve(gt_list, scores, args.name, '{}_{}'.format(args.phase, args.metric))
    refined_anomaly_maps = np.zeros_like(anomaly_maps)
    output_dir = "./test" 
    
    is_crf=False
    if is_crf:
        crf_masks = []
        for og, map in zip(test_imgs_list, anomaly_maps):
            map = get_seg(map, 24)
            crf_image = crf(og, map)
            crf_masks.append(crf_image[:,:,0])

    th = calculate_metrics(gt_mask_list, anomaly_maps)
    #calculate_metrics2(gt_mask_list, anomaly_maps, th)
    plot_scores_density(scores, ['max'], gt_list, args.name, '{}_{}'.format(args.phase, args.metric))
    if args.phase == 'test' and args.metric == 'msgms':
        for index, (i, ir, m, s, n) in enumerate(zip(test_imgs, recon_imgs, anomaly_maps, scores, names)):
            save_recon(i, ir, m, s, args.name, n, th)
    get_thresholds(gt_list, scores, args.name, args.phase, args.metric)
