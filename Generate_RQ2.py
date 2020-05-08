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

fig_num = 0
metrics_set = [['recall','pf'],['auc','pci_20'],['precision','ifa']]
for metrics in metrics_set:
    dfs = ['process','product','process+product']
    final_df = pd.DataFrame()
    learner_df = pd.DataFrame()
    learners = ['rf','lr','nb','svc']
    i = 0
    for learner in learners:
        for metric in metrics:
            data = []
            for df in dfs:
                file = pd.read_pickle('results/Performance/commit_guru_file_specific/' + df +'_700_' + learner + '_5_fold_5_repeat.pkl')
                if metric == 'ifa':
                    l = [item/100 for sublist in list(file[metric].values()) for item in sublist]
                else:
                    l = [item for sublist in list(file[metric].values()) for item in sublist]
                data.append(l)
            data_df = pd.DataFrame(data)
            data_df.index = [['P','C','P+C']]
            x = pd.melt(data_df.T)
            x.columns = ['Metric Type','Score']
            x['Evaluation Criteria'] = [metric]*x.shape[0]
            learner_df = pd.concat([learner_df,x])
        learner_df['learner'] = [learner]*learner_df.shape[0]
        final_df = pd.concat([final_df,learner_df])
        sns.set(style='whitegrid',font_scale=1)
        order = ["P", "C", "P+C"]
        g = sns.catplot(x="Metric Type", y="Score",col= 'learner' ,row="Evaluation Criteria",height=4,aspect=0.5,margin_titles=True,kind="box", 
                        order=order, data=final_df)
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]
        g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
        g.savefig('results/image/learner_' + fig_num + '.pdf')
    fig_num += 1
