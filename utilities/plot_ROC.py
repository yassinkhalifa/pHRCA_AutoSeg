import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

n_folds = 10
data_prepath = '../results/'
voutput_prepath = '../figures/'

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(12, 9))

acc = np.zeros((n_folds))
sens = np.zeros((n_folds))
spec = np.zeros((n_folds))

for i in range(n_folds):
    fold_ag_logits = np.load(os.path.join(data_prepath,'fold_{:0>2d}_aggregate_logits.npy'.format(i+1)))
    fold_ag_gtruth = np.load(os.path.join(data_prepath,'fold_{:0>2d}_aggregate_gtruth.npy'.format(i+1)))
    con_mat = confusion_matrix(fold_ag_gtruth, fold_ag_logits>0.5)
    TP = con_mat[1][1]
    TN = con_mat[0][0]
    FP = con_mat[0][1]
    FN = con_mat[1][0]
    acc[i] = (float (TP + TN)/float(TP + TN + FP + FN))
    sens[i] = (TP/float(TP + FN))
    spec[i] = (TN/float(TN + FP))
    fpr, tpr, _ = metrics.roc_curve(fold_ag_gtruth, fold_ag_logits)
    interp_tpr = np.interp(mean_fpr,fpr, tpr)
    interp_tpr[0] = 0
    tprs.append(interp_tpr)
    aucs.append(metrics.auc(fpr,tpr))

mean_acc = np.mean(acc)
std_acc = np.std(acc)
ci95_acc = st.norm.interval(alpha=0.95, loc=mean_acc, scale=st.sem(acc))
mean_sens = np.mean(sens)
std_sens = np.std(sens)
ci95_sens = st.norm.interval(alpha=0.95, loc=mean_sens, scale=st.sem(sens))
mean_spec = np.mean(spec)
std_spec = np.std(spec)
ci95_spec = st.norm.interval(alpha=0.95, loc=mean_spec, scale=st.sem(spec))

ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='#6F06C6', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ci95_auc = st.norm.interval(alpha=0.95, loc=mean_auc, scale=st.sem(aucs))
ax.plot(mean_fpr, mean_tpr, color='#057EFD', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#7AB6F5', alpha=.2, label=r'$\pm$ 1 std. dev. margin')

ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
ax.legend(loc ="lower right", fontsize='xx-large', facecolor='#81ADF0', edgecolor='none', framealpha=0.3)
plt.xlabel("False Positive Rate", fontsize='xx-large')
plt.ylabel("True Positive Rate", fontsize='xx-large')
plt.savefig(os.path.join(voutput_prepath, 'ss_avgROC.png'), format='png', dpi=600, bbox_inches='tight')
plt.show()