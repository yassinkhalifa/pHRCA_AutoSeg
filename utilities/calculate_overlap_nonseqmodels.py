import os
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set_style("darkgrid")

n_folds = 10
data_prepath = '../results/'
metadata_prepath = '../folds_metadata/'
Path(os.path.join(data_prepath, 'overlapratios')).mkdir(parents=True, exist_ok=True)
cfold_ratios = pd.DataFrame(columns=['swallow', 'overlap_ratio', 'fold'])
for i_fold in range(n_folds):
    print('Calculating swallowing ratios in fold {}/10\n'.format(i_fold+1))
    fold_pred_path = os.path.join(data_prepath, 'test_val')
    fold_gt_path = os.path.join(data_prepath, 'ref_val')
    sw_ratios = []
    fold_files = pd.read_csv(os.path.join(metadata_prepath, 'test_metadata_{}.csv'.format(i_fold+1)))
    fold_pred_labels = np.load(os.path.join(fold_pred_path, 'pred_fold_{:0>2d}.npy'.format(i_fold+1)))
    fold_gt_labels = np.load(os.path.join(fold_gt_path, 'ref_fold_{:0>2d}.npy'.format(i_fold+1)))
    frame_cnt = 0
    for file_cnt in range(len(fold_files)):
        frame_end = frame_cnt + int(fold_files['nb_frames'][file_cnt])
        if frame_end<=fold_pred_labels.shape[0]:
            gt_labels = np.squeeze(fold_gt_labels[frame_cnt:frame_end])
            pred_labels = np.squeeze(fold_pred_labels[frame_cnt:frame_end])
            d_gt = np.diff(gt_labels)
            n_peaks = np.max([np.sum(d_gt<0), np.sum(d_gt>0)])
            if n_peaks>0:
                start_idxs = np.where(d_gt==1)[0] + 1
                end_idxs = np.where(d_gt==-1)[0] + 1
                if len(start_idxs)>len(end_idxs):
                    end_idxs = np.append(end_idxs, len(pred_labels)-1)
                elif len(end_idxs)>len(start_idxs):
                    start_idxs = np.append(0, start_idxs)
                pred_prod = np.multiply(gt_labels, pred_labels)
                for i_s in range(n_peaks):
                    sw_overlap = np.sum(pred_prod[start_idxs[i_s]:end_idxs[i_s]])/(end_idxs[i_s]-start_idxs[i_s]+1)
                    sw_ratios.append(sw_overlap)
            frame_cnt = frame_cnt + int(fold_files['nb_frames'][file_cnt])
    sw_ratios = np.asarray(sw_ratios)
    fold_ratios = pd.DataFrame(columns=['swallow', 'overlap_ratio', 'fold'])
    fold_ratios['swallow'] = np.arange(0, len(sw_ratios))
    fold_ratios['overlap_ratio'] = sw_ratios
    fold_ratios['fold'] = 'fold{:0>2d}'.format(i_fold+1)
    cfold_ratios = cfold_ratios.append(fold_ratios)
cfold_ratios.to_csv(os.path.join(data_prepath, 'overlapratios', 'cfold_ratios.csv'), index=False)