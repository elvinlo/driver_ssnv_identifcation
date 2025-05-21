import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

from model import WeightedRandomForestClassifier, train_baseline_binary_rf
from simulation import generate_driver_mutation_data


def evaluate(model, X, y):
    pred      = model.predict(X)
    acc       = accuracy_score(y, pred)
    mae       = mean_absolute_error(y, pred)
    stderr_mae = np.std(np.abs(y - pred), ddof=1) / np.sqrt(len(y))
    return pred, (acc, mae, stderr_mae)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize: bool = False,
                          title: str = None,
                          filename: str = "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        # avoid division by zero
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype(float) / np.where(row_sums==0, 1, row_sums)

    fig, ax = plt.subplots()
    ax.imshow(cm)  
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    if title:
        ax.set_title(title)

    fmt = ".2f" if normalize else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def plot_bin_true_multi_pred(y_true_bin, y_pred_mc, mc_classes,
                             normalize=False,
                             title=None,
                             filename="confusion_matrix_bin_vs_mc.png"):
    """
    y_true_bin : array of 0/1
    y_pred_mc  : array of 1...K (same length)
    mc_classes : list/array of the multiclass labels (e.g. [1,2,…,10])
    """

    cm = np.zeros((2, len(mc_classes)), 
                  dtype=float if normalize else int)
    for i, t in enumerate([0, 1]):
        mask = (y_true_bin == t)
        for j, p in enumerate(mc_classes):
            cm[i, j] = np.sum(y_pred_mc[mask] == p)
        if normalize:
            s = cm[i].sum()
            if s > 0:
                cm[i] = cm[i] / s

    fig, ax = plt.subplots()
    cax = ax.imshow(cm, aspect="auto")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(mc_classes)))
    ax.set_xticklabels(mc_classes)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([0, 1])
    if title:
        ax.set_title(title)

    fmt = ".2f" if normalize else "d"
    for i in range(2):
        for j in range(len(mc_classes)):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center")

    fig.colorbar(cax, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)





# generate data
n_trees = 100
N = 6000

X, y = generate_driver_mutation_data(
    n=N,
    K=10,
    decay=0.80,
    random_state=42
)
y = y.astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=1
)

# fit and evaluate the multi-class models
base_params = dict(
    n_estimators   = n_trees,
    max_depth      = None,
    min_samples_split = 2,
    min_samples_leaf  = 1,
    random_state   = 1,
    n_jobs         = -1,
    max_features = 5
)

rf  = RandomForestClassifier(**base_params)
wrf = WeightedRandomForestClassifier(n_estimators=n_trees, max_features=5)

rf .fit(X_tr, y_tr)
wrf.fit(X_tr, y_tr)

y_pred_rf, metrics_rf  = evaluate(rf , X_te, y_te)
y_pred_wrf, metrics_wrf = evaluate(wrf, X_te, y_te)

header = f"{'Model':<18}  {'Accuracy':>9}  {'MAE':>8}  {'SE(MAE)':>9}"
print(header)
print("-" * len(header))
print(f"{'RandomForest':<18}  {metrics_rf [0]:9.3f}  {metrics_rf [1]:8.4f}  {metrics_rf [2]:9.4f}")
print(f"{'WeightedRF':<18}  {metrics_wrf[0]:9.3f}  {metrics_wrf[1]:8.4f}  {metrics_wrf[2]:9.4f}")

# plot the confusion matrices
classes = np.arange(1, 11)

plot_confusion_matrix(
    y_true   = y_te,
    y_pred   = y_pred_rf,
    classes  = classes,
    normalize= False,
    title    = "RandomForest Confusion Matrix",
    filename = "cm_rf.png"
)

plot_confusion_matrix(
    y_true   = y_te,
    y_pred   = y_pred_wrf,
    classes  = classes,
    normalize= False,
    title    = "WeightedRF Confusion Matrix",
    filename = "cm_wrf.png"
)

# conduct binarized evaluations
brf = train_baseline_binary_rf(
    X_tr, y_tr,
    threshold=7,
    n_estimators=n_trees,
    max_depth=None,
    min_samples_split=2,
    max_features=None,
    random_state=1
)

mask_te = (y_te >= 7) | (y_te == 1)
X_te_bin  = X_te[mask_te]
y_te_bin  = (y_te[mask_te] >= 7).astype(int)

# binarized the multi-class predictions (binning)
y_pred_brf = brf.predict(X_te_bin)
y_pred_wrf_bin = (y_pred_wrf[mask_te] >= 6).astype(int)
y_pred_rf_bin = (y_pred_rf[mask_te] >= 6).astype(int)

print("Baseline binary RF accuracy    :", accuracy_score(y_te_bin, y_pred_brf))
print("Weighted RF binarized accuracy :", accuracy_score(y_te_bin, y_pred_wrf_bin))
print("Vanilla RF binarized accuracy  :", accuracy_score(y_te_bin, y_pred_rf_bin))

# construct binarized confusion matrices
mc_classes = np.arange(1, 11)

plot_bin_true_multi_pred(
    y_true_bin = y_te_bin,
    y_pred_mc  = y_pred_rf[mask_te],
    mc_classes = mc_classes,
    normalize   = False,
    title       = "Vanilla RF: True(0/1) × Pred(1–10)",
    filename    = "cm_rf_bin_vs_mc.png"
)

plot_bin_true_multi_pred(
    y_true_bin = y_te_bin,
    y_pred_mc  = y_pred_wrf[mask_te],
    mc_classes = mc_classes,
    normalize   = False,
    title       = "Weighted RF: True(0/1) × Pred(1–10)",
    filename    = "cm_wrf_bin_vs_mc.png"
)