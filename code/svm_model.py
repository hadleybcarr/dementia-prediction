

import argparse
import os
import warnings
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    brier_score_loss, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay, confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import loguniform, uniform
from scipy import stats


from data_utils import get_dataloaders, SEQ_LEN, N_VITALS

def loader_to_arrays(loader):
    X_list, y_list = [], []
    for vitals, labels in loader:
        X_list.append(vitals.numpy())
        y_list.append(labels.numpy())
    X = np.concatenate(X_list, axis=0) #np.concatenate joins a series of existing arrays together
    y = np.concatenate(y_list, axis=0)
    return X, y.astype(int)

def featurize(X):
    N,T,C = X.shape
    t = np.arange(T, dtype=np.float32) #creates an array of evenly spaced numbers 
    t_centered = t - t.mean()
    denom = (t_centered ** 2).sum()

    feats = []
    for c in range(C):
        ch = X[:,:,c]
        mean = ch.mean(axis=1)
        std = ch.std(axis=1)
        mini = ch.min(axis=1)
        maxi = ch.max(axis=1)
        first = ch[:,0]
        last = ch[:, -1]
        ch_centered = ch - mean[:, None]
        slope = (ch_centered * t_centered).sum(axis=1) / denom
        feats.extend([mean, std, mini, maxi, first, last, slope])
        #Have to turn the values into linear values in order to be processed by the SVM; it is a linear model.s
    F = np.stack(feats,axis=1)
    return F

def svm_train(train_loader, test_loader, val_loader):
    X_train, y_train = loader_to_arrays(train_loader)
    X_val, y_val = loader_to_arrays(val_loader)
    X_test, y_test = loader_to_arrays(test_loader)

    print(f" Train: {X_train.shape}, labels: {y_train.shape}")
    print(f" Val: {X_val.shape}, labels: {y_val.shape}")
    print(f" test: {X_test.shape}, labels: {y_test.shape}")

    F_train = featurize(X_train)
    F_val = featurize(X_val)
    F_test = featurize(X_test)
    best = {"auc": -np.inf}
    for C in [0.1, 1.0, 10.0]:
        for gamma in ["scale", 0.01, 0.1]:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    C = C,
                    kernel = "rbf", 
                    gamma = gamma, 
                    class_weight="balanced", 
                    probability=True,
                    random_state=42,
                )),
            ])
            clf.fit(F_train, y_train)
            val_probs = clf.predict_proba(F_val)[:,1]
            auc = roc_auc_score(y_val, val_probs)
            print(f" C = {C}, gamma = {gamma}, AUC = {auc:.4}")
            if auc > best["auc"]:
                best = {"auc": auc, "C":C, "gamma":gamma, "clf":clf}

            clf = best["clf"]
            test_probs = clf.predict_proba(F_test)[:,1]
            test_pred = clf.predict(F_test)
            acc = accuracy_score(y_test, test_pred)
            area_under_curve = roc_auc_score(y_test, test_probs)
            print(f"Accuracy:{acc}")

            print(f"Area under curve:{area_under_curve}")
            
            #Graph AUC
            RocCurveDisplay.from_predictions(y_test,test_probs)
            plt.title('AUR-ROC')
            plt.savefig("svm_roc.png")

            print("\n Classification Report:")
            print(classification_report(y_test, test_pred, target_names=["control", "dementia"]))
            print("Confusion matrix")
            cm = confusion_matrix(y_test, test_pred)
            print(cm)

            #Graph confusion matrix
            confusion_matrix_graph = ConfusionMatrixDisplay(cm)
            confusion_matrix_graph.plot(cmap=plt.cm.Blues)
            plt.show()
            plt.savefig("svm_confusion_matrix.png")
