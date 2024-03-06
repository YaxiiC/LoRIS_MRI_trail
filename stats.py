import os
import pandas as pd
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report, roc_curve, confusion_matrix

def read_append_df(df, filename, metric):
    new_df = pd.read_csv('{}/test_{}/rec_results.csv'.format(filename, metric))
    df = pd.concat([df, new_df], ignore_index=True)
    return df

def plot_scores_density(scores, params, labels, name, fold):
    #print(test_df)
    #test_df["Anomaly_score1"] = anomaly_scores_ls[1]
    df = pd.DataFrame()
    df['scores'] = scores
    df['labels'] = labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    sns.kdeplot(data=df, x="scores", hue='labels', fill=True, common_norm=False, alpha=0.8, ax=ax1)
    ax1.set_xlabel("Anomaly score")
    ax1.set_title(params[0])
    plt.show()
    plt.savefig('./stats/{}/density_{}.png'.format(name, fold))
    plt.close()

def plot_roc_curve(gt_list, scores, name, fold):
    fpr, tpr, trh = roc_curve(gt_list, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = trh[optimal_idx]
    print("Threshold value is:", optimal_threshold)
    img_roc_auc = roc_auc_score(gt_list, scores)
    print('image ROCAUC: %.3f' % (img_roc_auc))
    plt.plot(fpr, tpr, label='Img_ROCAUC: %.3f' % (img_roc_auc))
    plt.legend(loc="lower right")
    plt.savefig('./stats/{}/ROCAUC_{}.png'.format(name, fold))
    plt.close()
    return optimal_threshold

def specificity_sensitivity(cm):
    tn = cm[0][0]
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    return specificity, sensitivity

def plt_thresholds(threshs, baccs, specs, senss, path):
    # Plot the curves
    plt.plot(threshs, baccs, label='Balanced acc', color='red')
    plt.plot(threshs, specs, label='Specificity', color='green')
    plt.plot(threshs, senss, label='Sensitivity', color='darkgreen')

    # Add legend
    plt.legend()

    # Add titles and labels
    plt.title('Thresh scores')
    plt.xlabel('tresholds')
    plt.ylabel('Metric')

    # Show the plot
    plt.show()
    plt.savefig('./stats/{}/tresh.png'.format(path))
    plt.close()

def get_thresholds(gt_list, scores, path, single):
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
    
    threshs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.76, 0.8, 0.85, 0.9, 0.95]
    baccs, specs, senss = [], [], []
    
    max_bacc, max_spec, max_sec = 0, 0, 0
    
    with open('./stats/{}/res.txt'.format(path), 'w') as f:
        if not single:
            f.write('ND interval: {}-{} ||| D interval {}-{}\n'.format(min_nd, max_nd, min_d, max_d))
        for t in threshs:
            print('Threshold: {}'.format(t))
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for l, s in zip(gt_list, scores):
                #print('-- {}-{} --'.format(c, p))
                if l == 1 and s >= t:
                    tp += 1
                elif l == 0 and s < t:
                    tn += 1
                elif l == 1 and s < t:
                    fn += 1
                elif l == 0 and s >= t:
                    fp += 1
            #print('TN = {}, FP = {}'.format(tn, fp))
            #print('FN = {}, TP = {}'.format(fn, tp))
            cr = classification_report(gt_list, [int(p>=t) for p in scores], target_names=['ND', 'D'])
            f.write(cr)
            cm=confusion_matrix(gt_list, [int(p>=t) for p in scores])
            f.write(str(cm))
            specificity, sensitivity = specificity_sensitivity(cm)
            if not single:
                f.write('\nSpecificity: {}, Sensitivity: {}, B Accuracy: {}\n'.format(specificity, sensitivity, (specificity+sensitivity)/2))
                print('Specificity: {}, Sensitivity: {}, B Accuracy: {}\n'.format(specificity, sensitivity, (specificity+sensitivity)/2))
                f.write('---------------------------------------------------------------------------------------------\n')
            baccs.append((specificity+sensitivity)/2)
            specs.append(specificity)
            senss.append(sensitivity)
            if (specificity+sensitivity)/2 > max_bacc:
                max_bacc = (specificity+sensitivity)/2
                max_spec = specificity
                max_sens = sensitivity
    if not single:
        plt_thresholds(threshs, baccs, specs, senss, path)
    return max_bacc, max_spec, max_sens
def get_stats(gt_list, scores, path, single, thresh, fold):
    with open('./stats/{}/best_tr_res_{}.txt'.format(path, fold), 'w') as f:
        print('Threshold: {}'.format(thresh))
        f.write('Threshold: {}'.format(thresh))
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for l, s in zip(gt_list, scores):
            #print('-- {}-{} --'.format(c, p))
            if l == 1 and s >= thresh:
                tp += 1
            elif l == 0 and s < thresh:
                tn += 1
            elif l == 1 and s < thresh:
                fn += 1
            elif l == 0 and s >= thresh:
                fp += 1
        cr = classification_report(gt_list, [int(p>=thresh) for p in scores], target_names=['ND', 'D'])
        f.write(cr)
        cm=confusion_matrix(gt_list, [int(p>=thresh) for p in scores])
        f.write(str(cm))
        specificity, sensitivity = specificity_sensitivity(cm)
        if single:
            f.write('\nSpecificity: {}, Sensitivity: {}, B Accuracy: {}\n'.format(specificity, sensitivity, (specificity+sensitivity)/2))
            f.write('---------------------------------------------------------------------------------------------\n')
            print('Specificity: {}, Sensitivity: {}, B Accuracy: {}\n'.format(specificity, sensitivity, (specificity+sensitivity)/2))
        bacc = ((specificity+sensitivity)/2)
        spec = specificity
        sens = sensitivity
    return bacc, spec, sens

def print_roc(gt_list, scores, name, fold):
    fpr, tpr, _ = roc_curve(gt_list, scores)
    img_roc_auc = roc_auc_score(gt_list, scores)
    print('{} ROCAUC: {}'.format(name, img_roc_auc))
    with open('./stats/{}/stats_{}.txt'.format(args.name, fold), 'a', newline='') as f:
        f.write('{} ROCAUC: {}\n'.format(name+str(fold), img_roc_auc))
    return img_roc_auc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LRIAD anomaly detection stats')
    parser.add_argument('--name', default='name')
    parser.add_argument('--metric', default='dd')
    args = parser.parse_args()
    df = pd.DataFrame()
    if not os.path.isdir('./stats/{}'.format(args.name)):
        os.mkdir('./stats/{}'.format(args.name))
    dirlist = glob.glob('./runs/*')
    mean_auroc = 0
    num_folds = 0
    mean_bacc = 0
    mean_spec = 0
    mean_sens = 0
    with open('./stats/{}/stats.txt'.format(args.name), 'w', newline='') as f:
        f.write('------ Stats for {} --------\n'.format(args.name))
    
    
    for d in dirlist:
        new_df = pd.DataFrame()
        print(d)
        if args.name in d:
            if os.path.isdir('{}/test_{}'.format(d, args.metric)):
                df = read_append_df(df, d, args.metric)
                new_df = read_append_df(new_df, d, args.metric)
                single_labels = new_df['labels']
                single_scores = new_df['scores']
                max_anomaly_score = single_scores.max()
                min_anomaly_score = single_scores.min()
                single_scores = (single_scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
                single_labels = np.asarray(single_labels)
                temp_roc = print_roc(single_labels, single_scores, args.name, num_folds)
                fold_bacc, fold_spec, fold_sens = get_thresholds(single_labels, single_scores, args.name, True)
                tr = plot_roc_curve(single_labels, single_scores, args.name, num_folds)
                plot_scores_density(single_scores, ['max'], single_labels, args.name, num_folds)
                bacc, spec, sens = get_stats(single_labels, single_scores, args.name, True, tr, num_folds)
                mean_bacc += fold_bacc
                mean_spec += fold_spec
                mean_sens += fold_sens
                mean_auroc +=temp_roc
                num_folds += 1
                with open('./stats/{}/stats.txt'.format(args.name), 'a', newline='') as f:
                    f.write('{} bacc: {}, spec {}, sens: {}\n'.format(args.name+str(num_folds), fold_bacc, fold_spec, fold_sens))
                    f.write('---------------------------------------------------------------------------------------------\n')
            else:
                print('{} not completed'.format(d))
    mean_auroc /= num_folds
    mean_bacc /= num_folds
    mean_spec /= num_folds
    mean_sens /= num_folds
    print('Mean ROCAUC: {}'.format(mean_auroc))
    print('Mean bacc: {}, spec {}, sens: {}\n'.format(mean_bacc, mean_spec, mean_sens))
    with open('./stats/{}/stats.txt'.format(args.name), 'a', newline='') as f:
        f.write('Mean ROCAUC: {}\n'.format(mean_auroc))
        f.write('Mean bacc: {}, spec {}, sens: {}\n'.format(mean_bacc, mean_spec, mean_sens))
    
    labels = df['labels']
    scores = df['scores']
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
    labels = np.asarray(labels)
    opt_tresh = plot_roc_curve(labels, scores, args.name, 'all')
    plot_scores_density(scores, ['max'], labels, args.name, 'all')
    get_thresholds(labels, scores, args.name, False)
    get_stats(labels, scores, args.name, False, opt_tresh, 'all')