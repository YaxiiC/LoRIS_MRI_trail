import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns
import pandas as pd
import os
import random
import time
import argparse
import csv
from tqdm import tqdm

from utils.gen_mask import gen_mask
from utils.funcs import specificity_sensitivity, EarlyStop

from tensorflow import keras
from tensorflow.keras import layers, activations

from datasets.LoRIS_.datasets.tf_dataset_ import CustomDataGen, denormalize, normalize
from losses.tf_criterions import which_loss, COMBINED_W, directional_difference
from utils.utils import AverageMeter, generate_run, save_args
from models.tf_unet import AutoEncoder
from utils.plots import save_image

@tf.function
def val_step(model, inputs):
    outputs = [model(x) for x in inputs]
    return outputs
    
def val(args, model, test_gen, loss_fn, epoch, save, path):
    losses = AverageMeter()
    l2_losses = AverageMeter()
    gms_losses = AverageMeter()
    ssim_losses = AverageMeter()
    for step, (data, _, _, _) in enumerate(test_gen):
        k_value = random.sample(args.k_values, 1)
        Ms_generators = []
        Mss = []
        Ms_generator = gen_mask(k_value, 3, args.img_size)
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
        
        outputs = val_step(model, inputs)
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

        loss_value = loss_fn(data, output)
        
        l2_losses.update(tf.get_static_value(loss_value[3]), len(data))
        gms_losses.update(tf.get_static_value(loss_value[1]), len(data))
        ssim_losses.update(tf.get_static_value(loss_value[2]), len(data))
        losses.update(tf.get_static_value(loss_value[0]), len(data))
    save_image(output[0], output[1], output[2], output[3], epoch, 0, path, 'val_batch', ['R0', 'R1', 'R2', 'R3'])
    save_image(data[0], output[0], data[1], output[1], epoch, 0, path, 'validation', ['Original 0', 'Reconstructed 0', 'Original 1', 'Reconstructed 1'])
    plt.close('all') 
    print('Valid Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))
    return [losses, gms_losses, ssim_losses, l2_losses, loss_value[0]]

@tf.function
def train_step(model, inputs):
    outputs = [model(x) for x in inputs]
    return outputs

def train_custom(args, model, train_gen, loss_fn, optimizer, activation, epoch, save, path):
    # Iterate over the batches of the dataset.
    l2_losses = AverageMeter()
    gms_losses = AverageMeter()
    ssim_losses = AverageMeter()
    total_losses = AverageMeter()
    for step, (data, _, _, _) in enumerate(tqdm(train_gen)):
        k_value = random.sample(args.k_values, 1)
        Ms_generators = []
        Mss = []
        Ms_generator = gen_mask(k_value, 3, args.img_size)
        Ms_generators.append(Ms_generator)
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
        with tf.GradientTape() as tape:
            outputs = train_step(model, inputs)
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

            loss_value = loss_fn(data, output)

        grads = tape.gradient(loss_value[0], model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        l2_losses.update(tf.get_static_value(loss_value[3]), len(data))
        gms_losses.update(tf.get_static_value(loss_value[1]), len(data))
        ssim_losses.update(tf.get_static_value(loss_value[2]), len(data))
        total_losses.update(tf.get_static_value(loss_value[0]), len(data))
    if epoch % 10 == 0 or epoch==1 and step == 0:
        save_image(data[0], inputs[0][0], data[1], inputs[1][0], epoch, 0, path, 'input', ['Original 0', 'Input 0', 'Original 1', 'Input 1'])
        save_image(data[0], output[0], data[1], output[1], epoch, 0, path, 'train', ['Original 0', 'Reconstructed 0', 'Original 1', 'Reconstructed 1'])
    if save or epoch % 10 == 0:
        save_model(model, path, 'training')
        save_image(output[0], output[1], output[2], output[3], epoch, 0, path, 'batch', ['R0', 'R1', 'R2', 'R3'])
    plt.close('all') 
    return [total_losses, gms_losses, ssim_losses, l2_losses, loss_value[0]]

def save_model(model, path ,name):
    model.save('{}/models/{}_model'.format(path, name))
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RIAD anomaly detection')
    parser.add_argument('--epochs', type=int, default=100, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--name', default='run')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--cv_fold', default=None)
    parser.add_argument('--k_values', type=int, nargs='+', default=[2, 4, 8, 16])
    args = parser.parse_args()
    
    print(tf.__version__)
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
    
    path = generate_run(args.name + '_' +str(int(time.time())))
    save_args(args, path)
    
    # -- set --
    loss = 'COMBINED'
    activation = 'tanh'
    lam = 0.75

    # Build model

    loss_fn   = COMBINED_W(lam) if loss=='COMBINED_W' else which_loss(loss) 
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model     = AutoEncoder(activation)
    model.compile(optimizer = optimizer, loss = loss_fn)
    data_path = '../anomaly' if args.cv_fold is None else '../anomaly_{}'.format(args.cv_fold)
    train_gen = CustomDataGen(data_path, class_name='sqr', is_train=True, phase='train', resize=args.img_size, batch_size=args.batch_size)
    val_gen = CustomDataGen(data_path, class_name='sqr', is_train=True, phase='valid', resize=args.img_size, batch_size=args.batch_size)

    early_stop = EarlyStop(patience=args.patience, path=path)

    epoch_time = AverageMeter()
    begin_time = time.time()
    start_time = time.time()
    save = False
    
    header = ["Epoch", "Train Loss", "Train MSGMS", "Train SSIM", "Train L2", "Val Loss", "Val MSGMS", "Val SSIM", "Val L2"]
    with open('{}/train_log.csv'.format(path), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    for epoch in range(1, args.epochs + 1):
        if epoch == args.epochs:
            save = True
        losses = train_custom(args, model, train_gen, loss_fn, optimizer, activation, epoch, save, path)
        print('Train Epoch: {} L2_Loss: {:.6f} GMS_Loss: {:.6f} SSIM_Loss: {:.6f} in {}'.format(epoch, losses[3].avg, losses[1].avg, losses[2].avg, time.time() - start_time))
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        train_gen.on_epoch_end()
        
        val_losses = val(args, model, val_gen, loss_fn, epoch, save, path)
        with open('{}/train_log.csv'.format(path), 'a', newline='') as file:
            line = [epoch, losses[0].avg, losses[1].avg, losses[2].avg, losses[3].avg, val_losses[0].avg, val_losses[1].avg, val_losses[2].avg, val_losses[3].avg]
            writer = csv.writer(file)
            writer.writerow(line)
        if (early_stop(val_losses[4], model, optimizer, '')):
            save_model(model, path, 'final_{}'.format(epoch))
            break
        plt.close('all')
    print('Total time occurred for {} epochs: {}'.format(epoch, time.time() - begin_time))