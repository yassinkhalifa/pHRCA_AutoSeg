import os
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set_style("white")

voutput_prepath = '../figures/'

model1_overlaps = pd.read_csv(os.path.join('../run_5_spectSig_062720/results', 'overlapratios', 'cfold_ratios.csv'))
model2_overlaps = pd.read_csv(os.path.join('../run_6_rawSig_070720/results', 'overlapratios', 'cfold_ratios.csv'))
model3_overlaps = pd.read_csv(os.path.join('../run_7_spectSigfcn_081020/results', 'overlapratios', 'cfold_ratios.csv'))
model4_overlaps = pd.read_csv(os.path.join('../run_8_ssVGG16_081320/results', 'overlapratios', 'cfold_ratios.csv'))
model5_overlaps = pd.read_csv(os.path.join('../run_9_rwVGG16_081320/results', 'overlapratios', 'cfold_ratios.csv'))
model6_overlaps = pd.read_csv(os.path.join('../run_10_fcVGG16_081620/results', 'overlapratios', 'cfold_ratios.csv'))
model7_overlaps = pd.read_csv(os.path.join('../run_11_ssResVGG16_091820/results', 'overlapratios', 'cfold_ratios.csv'))
model8_overlaps = pd.read_csv(os.path.join('../run_12_rwResVGG16_091820/results', 'overlapratios', 'cfold_ratios.csv'))
model9_overlaps = pd.read_csv(os.path.join('../run_13_fcResVGG16_092120/results', 'overlapratios', 'cfold_ratios.csv'))

fig, ax = plt.subplots(num=None, figsize=(15, 12), facecolor='w', edgecolor='k')
sns.lineplot(x="fold", y="overlap_ratio", data=model1_overlaps, label="2D shallow CRNN + Spectrogram input", legend="brief", err_style="bars", ax=ax)
sns.lineplot(x="fold", y="overlap_ratio", data=model2_overlaps, label="1D shallow CRNN + Raw signals input", legend="brief", err_style="bars", ax=ax)
sns.lineplot(x="fold", y="overlap_ratio", data=model3_overlaps, label="5-layer NN + Spectrogram input", legend="brief", err_style="bars", ax=ax)
plt.ylim(-0.3,1.2)
plt.xlabel("Fold", fontsize='xx-large')
plt.ylabel("Overlap Ratio", fontsize='xx-large')
plt.legend(loc=1, fontsize='x-large')
ax.xaxis.grid(False)
plt.savefig(os.path.join(voutput_prepath, 'testing_overlapping_full.png'), format='png', dpi=600, bbox_inches='tight')
plt.show()