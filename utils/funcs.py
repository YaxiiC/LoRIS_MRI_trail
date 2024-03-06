import numpy as np
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def save_res(names, gt_list, scores, path, phase, metric):
    df = pd.DataFrame()
    df['names'] = names
    df['labels'] = gt_list
    df['scores'] = scores
    df.to_csv('./runs/{}/{}_{}/rec_results.csv'.format(path, phase, metric))

def get_thresholds(gt_list, scores, path, phase, metric):
    max_nd = 0
    min_nd = 1
    max_d = 0
    min_d = 1
    
    for l, s in zip(gt_list, scores):
        if l == 0:
            if max_nd < s:
                max_nd = s
            if min_nd > s:
                min_nd = s
        else:
            if max_d < s:
                max_d = s
            if min_d > s:
                min_d = s
    
    threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    with open('./runs/{}/{}_{}/res.txt'.format(path, phase, metric), 'w') as f:
        f.write('ND interval: {}-{} ||| D interval {}-{}\n'.format(min_nd, max_nd, min_d, max_d))
        for t in threshs:
            f.write('Threshold: {}'.format(t))
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for l, s in zip(gt_list, scores):
                if l == 1 and s >= t:
                    tp += 1
                elif l == 0 and s < t:
                    tn += 1
                elif l == 1 and s < t:
                    fn += 1
                elif l == 0 and s >= t:
                    fp += 1

            cr = classification_report(gt_list, [int(p>=t) for p in scores], target_names=['ND', 'D'])

            f.write(cr)
            cm=confusion_matrix(gt_list, [int(p>=t) for p in scores])
            f.write(str(cm))

            specificity, sensitivity = specificity_sensitivity(cm)
            f.write('Specificity: {}, Sensitivity: {}, B Accuracy: {}\n'.format(specificity, sensitivity, (specificity+sensitivity)/2))
            
            f.write('---------------------------------------------------------------------------------------------\n')

def specificity_sensitivity(cm):
    tn = cm[0][0]
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    return specificity, sensitivity

def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

class EarlyStop():
    def __init__(self, patience=20, delta=0, path='name'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, optimizer, log):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print((f'EarlyStopping counter: {self.counter} out of {self.patience}'))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_best(model)

        return self.early_stop
        
    def save_best(self, model):
        model.save('{}/models/best_model'.format(self.path))