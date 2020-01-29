from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import io, stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

from my_logger import get_my_logger

logger = get_my_logger(Path(__file__).name)


def load_data(data_name):
    path = Path(f'data/input/{data_name}_data.mat')
    data = io.loadmat(path)
    return data['training_data'], data['test_data'], data['training_labels'].reshape(-1,)


def tr_va_split(x_train, y_train, n_va, shuffle=False):
    """
    x_train: np.array
        Features of all training dataset

    y_train: np.array
        Label of all training dataset

    n_va: float or int
        If float, shoud be 0.0 to 1.0 and represent the
        proportion of validation dataset. If int, represent the
        absolute number of training dataset

    shuffle: boolean (default=False)
        Whether or not to shuffle the data before splitting
    """
    n_samples = x_train.shape[0]
    if n_va < 1:
        n_va = int(n_samples * n_va)  # from proportion to absolute number
    else:
        n_va = n_va

    if shuffle:
        ind_all = np.random.permutation(n_samples)  # shuffle index
    else:
        ind_all = np.arange(n_samples)  # not shuffle index

    yield ind_all[n_va:], ind_all[:n_va]


def kfold_split(x_train, y_train, n_fold, shuffle=False):
    n_samples = x_train.shape[0]
    # shuffle
    if shuffle:
        ind_all = np.random.permutation(n_samples)
    else:
        ind_all = np.arange(n_samples)

    f_sizes = [(n_samples + i) // n_fold for i in range(n_fold)]
    end = 0
    for f_size in f_sizes:
        start, end = end, end + f_size
        ind_tr = np.concatenate([ind_all[end:], ind_all[:start]], axis=0)
        ind_va = ind_all[start:end]
        yield ind_tr, ind_va


def plot_cm(y_va, y_va_pred, path):
    cm = confusion_matrix(y_va, y_va_pred, normalize='true')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    fig.savefig(path)
    return fig, ax


def plot_accs(accs, data_name, path, xlabel, xlim=None, log_scale=False):
    accs_df = pd.DataFrame(accs)
    fig, ax = plt.subplots(figsize=(7, 4))
    accs_df.plot.line(marker='x', lw=0.5, ax=ax)
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(0, 1)
    ax.set_title(str(path))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy')
    if log_scale:
        ax.set_xscale('log')
    ax.grid()
    fig.tight_layout()
    fig.savefig(path)


class Trainer:
    def __init__(self, params, x_train, x_test, y_train, verbose):
        self.params = params
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.verbose = verbose
        self.y_train_pred = np.zeros_like(y_train)
        self.y_test_pred = pd.DataFrame()
        self.scores = defaultdict(list)
        self.scores_mean = {}

    def train(self, x_tr, y_tr):
        self.model.fit(x_tr, y_tr)

    def predict(self, x_tr, x_va, y_tr, y_va, ind_va, n_fold, i_fold):
        y_tr_pred = self.model.predict(x_tr)
        y_va_pred = self.model.predict(x_va)
        self.y_train_pred[ind_va] = y_va_pred
        self.y_test_pred[i_fold] = self.model.predict(self.x_test)

        tr_acc = accuracy_score(y_tr, y_tr_pred)
        va_acc = accuracy_score(y_va, y_va_pred)
        self.scores['tr_acc'].append(tr_acc)
        self.scores['va_acc'].append(va_acc)

        if self.verbose:
            logger.info(f'[fold{i_fold}]\ttr:{tr_acc:.5f}\tva:{va_acc:.5f}')

    def validate(self, split, n_fold=1, n_va=None, n_tr=None, shuffle=False):
        if split == 'holdout':
            cv = tr_va_split(self.x_train, self.x_test, n_va, shuffle=shuffle)
        elif split == 'kfold':
            cv = kfold_split(self.x_train, self.x_test, n_fold, shuffle=shuffle)
        else:
            raise Exception(f'invalid variable name of split: {split}')

        for i_fold, (ind_tr, ind_va) in enumerate(cv):
            x_tr, x_va = self.x_train[ind_tr[:n_tr]], self.x_train[ind_va]
            y_tr, y_va = self.y_train[ind_tr[:n_tr]], self.y_train[ind_va]

            self.model = SVC(**self.params)
            self.train(x_tr, y_tr)
            self.predict(x_tr, x_va, y_tr, y_va, ind_va, n_fold, i_fold)

        for key in self.scores.keys():
            self.scores_mean[key] = np.mean(self.scores[key])

        self.y_test_pred = stats.mode(self.y_test_pred, axis=1)[0].ravel()
