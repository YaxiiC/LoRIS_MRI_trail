import torch
import cv2
import os
from torchvision import transforms
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image


IMG_WIDTH    = 256
IMG_HEIGHT   = 256
IMG_CHANNELS = 3
size = (IMG_WIDTH,IMG_HEIGHT)

def read_bb(filename):
    with open(filename) as file:
        yolo_coords = file.readline()
        yolo_coords = yolo_coords[2:]
        floats = [float(x) for x in yolo_coords.split()]
        return xywh2xyxy(floats, 256, 256)

def get_box_fname(imname):
    file_name_without_extension = os.path.splitext(os.path.basename(imname))[0]
    return '{}.txt'.format(file_name_without_extension)
    

def xywh2xyxy(yolo_coords, width=256, height=256):
    if len(yolo_coords) == 4:
        x, y, w, h  = yolo_coords
        x1 = (x - w / 2)*width
        y1 = (y - h / 2)*height
        x2 = (x + w / 2)*width
        y2 = (y + h / 2)*height
        coords = [x1, y1, x2, y2]
        coords = [int(c) for c in coords]
    elif len(yolo_coords) == 5:
        x, y, w, h  = yolo_coords[:4]
        x1 = (x - w / 2)*width
        y1 = (y - h / 2)*height
        x2 = (x + w / 2)*width
        y2 = (y + h / 2)*height
        coords = [x1, y1, x2, y2]
        coords = [int(c) for c in coords]
    else: 
        coords = ''
    return coords


def mask_image(image, mask_coords = [110, 150, 150, 110]):
    mask_color = (0, 0, 0) 
    x, y, w, z = mask_coords
    copy_image = image.copy()
    if tf.is_tensor(image):                               
        copy_image = copy_image.numpy()
    cv2.rectangle(copy_image, (x, y), (w, z), mask_color, -1) 
    if tf.is_tensor(image):                                  
        copy_image = tf.constant(copy_image)
    return copy_image


''' normalization functions '''
def normalize(X):
       
    # normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    a = []
    for img in X:
        tensor = transform(img)
        tensor = tensor.unsqueeze(0)
        a.append(tensor)
    X_torch = torch.cat(a)
    X_array = np.array(X_torch)
    X_tf = tf.transpose(X_array, perm=[0, 2, 3, 1])        
    return X_tf

def denormalize(images):
    return (tf.round(tf.multiply(images + 1, 127.5))/255).numpy()

CLASS_NAMES = ['sqr']

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self,
                 dataset_path='../anomaly_0',
                 class_name='sqr',
                 is_train=True,
                 phase='train',
                 resize=256,
                 batch_size=4,
                 fold=0
                 ):
        
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.batch_size = batch_size
        self.phase = phase
        # load dataset
        self.x, self.y, self.coord, self.mask = self.load_dataset_folder(fold)
        self.indexes = np.arange(len(self.x))

        self.transform_x = transforms.Compose([
             transforms.Resize(resize, Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
        self.transform_mask = transforms.Compose(
            [transforms.Resize(resize, Image.NEAREST),
             transforms.ToTensor()])
            
    def normalize(self, img):
        a = []
        for x in img:
            tensor = self.transform_x(x)
            tensor = tensor.unsqueeze(0)
            a.append(tensor)
        X_torch = torch.cat(a)
        X_array = np.array(X_torch)
        X_tf = tf.transpose(X_array, perm=[0, 2, 3, 1])        
        return X_tf


    def load_dataset_folder(self, fold):
        phase = self.phase
        x, y, boxes, mask = [], [], [], []
        img_dir = os.path.join('../anomaly_{}'.format(fold), self.class_name, phase)
        box_dir = os.path.join('../anomaly_{}'.format(fold), self.class_name, 'boxes')
        gt_dir = os.path.join('../anomaly_{}'.format(fold), self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        
        for img_type in img_types:
            if self.is_train and not img_type == 'nd':
                continue
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            x.extend(img_fpath_list)

            box_fpath_list = map(get_box_fname, img_fpath_list)
            box_fpath_list = [os.path.join(box_dir, f) for f in box_fpath_list]
            boxes.extend(box_fpath_list)

            if img_type == 'nd':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                
        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(boxes), list(mask)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.is_train == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        x, y, coord, mask = self.x, self.y, self.coord, self.mask
        xs = []
        ys = []
        coords = []
        names = []
        masks = []
        for i in indexes:
            img = Image.open(x[i]).convert('RGB')#TODO:check read image
            if y[i] == 0:
                msk = torch.zeros([1, self.resize, self.resize], dtype=torch.float32)
            else:
                msk = Image.open(mask[i])
                msk = self.transform_mask(msk).float()
                if msk.ndim == 2:
                    msk = msk.unsqueeze(0)  # Ensure single-channel mask

            co = read_bb(coord[i])
            xs.append(img)
            coords.append(co)
            masks.append(msk)
            ys.append(y[i])
            names.append(x[i].split('/')[-1].split('.')[0])
    
        xs = self.normalize(xs)
        return xs, ys, coords, names, masks
    
    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))
