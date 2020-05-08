import pandas as pd
import numpy as np
import math
import pickle

from scipy import stats
import scipy.io
from scipy.spatial.distance import pdist
from scipy.linalg import cholesky
from scipy.io import loadmat

import matlab.engine as engi
import matlab as mat

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,roc_auc_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from pyearth import Earth
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

from src import SMOTE
from src import CFS
from src import metrices_V2 as metrices

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import copy
import traceback
from pathlib import Path

# Venn diag
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_pickle('results/Performance/commit_guru_file_specific/process+product_700_rf_cross.pkl')
df = df['featue_importance']
df = pd.DataFrame.from_dict(df,orient='index')
df.columns = ['AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict',
       'AvgEssential', 'AvgLine', 'AvgLineBlank', 'AvgLineCode',
       'AvgLineComment', 'CountClassBase', 'CountClassCoupled',
       'CountClassCoupledModified', 'CountClassDerived',
       'CountDeclClassMethod', 'CountDeclClassVariable',
       'CountDeclInstanceMethod', 'CountDeclInstanceVariable',
       'CountDeclMethod', 'CountDeclMethodAll', 'CountDeclMethodDefault',
       'CountDeclMethodPrivate', 'CountDeclMethodProtected',
       'CountDeclMethodPublic', 'CountLine', 'CountLineBlank', 'CountLineCode',
       'CountLineCodeDecl', 'CountLineCodeExe', 'CountLineComment',
       'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe',
       'MaxCyclomatic', 'MaxCyclomaticModified', 'MaxCyclomaticStrict',
       'MaxEssential', 'MaxInheritanceTree', 'MaxNesting',
       'PercentLackOfCohesion', 'PercentLackOfCohesionModified',
       'RatioCommentToCode', 'SumCyclomatic', 'SumCyclomaticModified',
       'SumCyclomaticStrict', 'SumEssential', 'la', 'ld', 'lt',
       'age', 'ddev', 'nuc', 'own', 'minor', 'ndev',
       'ncomm', 'adev', 'nadev', 'avg_nddev',
       'avg_nadev', 'avg_ncomm', 'ns', 'exp', 'sexp',
       'rexp', 'nd', 'sctr']
df_stats = pd.DataFrame(df.quantile([.25, .5, .75]))
df_stats = df_stats.T
df_stats.columns = ['25th','50th','75th']
df_stats = df_stats.sort_values(by = ['50th'])
x = range(0,66)
y1 = np.array(df_stats['25th'].values.tolist())
y2 = np.array(df_stats['50th'].values.tolist())
y3 = np.array(df_stats['75th'].values.tolist())
fig, ax = plt.subplots(figsize=(15,5))
# ax.set_ylim(0,0.25)
# ax.set_xlim(66, 0)
ax.set_xticks(x)
ax.plot(y2,linestyle='-', color='b', linewidth='2')

_x_tick = ['**nd', '**ns', 'CountDeclMethodDefault', 'CountClassDerived', 'AvgEssential', 
'CountClassBase', 'CountDeclMethodProtected', 'MaxInheritanceTree', 'AvgLineBlank', 
'CountDeclClassMethod', 'MaxEssential', 'AvgCyclomaticModified', 'AvgLineComment', 
'AvgCyclomatic', 'CountDeclMethodPrivate', 'AvgCyclomaticStrict', 'CountDeclClassVariable', 
'MaxNesting', 'MaxCyclomaticModified', 'MaxCyclomatic', 'CountDeclInstanceVariable', 
'MaxCyclomaticStrict', 'CountClassCoupled', 'CountClassCoupledModified', 
'PercentLackOfCohesionModified', 'CountDeclMethodPublic', 'CountDeclInstanceMethod', 
'PercentLackOfCohesion', 'SumEssential', 'AvgLineCode', 'SumCyclomaticModified', 
'CountDeclMethod', 'AvgLine', 'SumCyclomatic', 'SumCyclomaticStrict', 
'CountDeclMethodAll', 'RatioCommentToCode', 'CountLineComment', 
'CountStmtExe', 'CountLineBlank', 'CountStmtDecl', 'CountSemicolon', 
'CountLineCodeExe', 'CountLineCodeDecl', 'CountStmt', 'CountLineCode', 
'CountLine', '**lt', '**la', '**ld', '**sctr', '**sexp', '**exp', '**avg_nddev', '**ddev', '**ndev', 
'**own', '**age', '**nuc', '**adev', '**minor', '**ncomm', '**nadev', '**rexp', '**avg_ncomm', '**avg_nadev']

ax.set_xticklabels(_x_tick)

colors = []
process = ['**la', '**ld', '**lt',
       '**age', '**ddev', '**nuc', '**own', '**minor', '**ndev',
       '**ncomm', '**adev', '**nadev', '**avg_nddev',
       '**avg_nadev', '**avg_ncomm', '**ns', '**exp', '**sexp',
       '**rexp', '**nd', '**sctr']

for tick in _x_tick:
    if tick in process:
        colors.append('blue')
    else:
        colors.append('black')

for xtick, color in zip(ax.get_xticklabels(), colors):
    xtick.set_color(color)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.fill_between(x, y2 - y1, y2 + y3, alpha=0.2,color='red')
plt.xticks(fontsize=12,rotation=90)
plt.yticks(fontsize=12,rotation=90)
plt.grid(b=None)
plt.savefig('results/image/process+product_feature.pdf',dpi=600,bbox_inches='tight', pad_inches=0.3)