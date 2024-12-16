"""
Implements thresholding strategies.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
import numpy as np
thresholds = [x/100.0 for x in range(1, 90, 1)]

def lco(preds, Y_train):
    """
    implements Read et al.
    :param preds:
    :param Y_train:
    :return:
    """
    t = which_threshold(preds, Y_train)
    A = np.where(preds >= t, 1, 0)
    A = correct_all_zero_rows(A, preds)
    return A

def which_threshold(preds, Y_train):
    lcard = np.sum(Y_train) / Y_train.shape[0]
    nrow = preds.shape[0]
    preds_lcard = [np.sum(preds >= thresholds[i])/nrow for i in range(len(thresholds))]
    ind =  np.argmin(abs(lcard - preds_lcard))
    return thresholds[ind]


### implements class-mass normalization method of Zhu et al., 2003
def cmn(preds, Y_train):
    nrow, ncol = preds.shape
    A = np.zeros((nrow, ncol))
    Y_train_p = Y_train.mean(axis=0)
    fu = preds.sum(axis=0)
    for j in range(ncol):
        class_j = preds[:, j]
        fuc = nrow - fu[j]
        A[:, j] = np.sign(((Y_train_p[j] * class_j) / fu[j]) - (1 - Y_train_p[j]) * ((1 - class_j) / fuc))
    A[A == -1.] = 0
    A = correct_all_zero_rows(A, preds)
    return A

### thresholding at 0
def atzero(preds, Y_train=None):
    A = np.where(preds >= 0, 1, 0)
    A = correct_all_zero_rows(A, preds)
    return A

### thresholding at 0.5
def basic(preds, Y_train=None):
    A = np.where(preds >= 0.5, 1, 0)
    A = correct_all_zero_rows(A, preds)
    return A

def correct_all_zero_rows(A, preds):
    zero_rows = np.all(A == 0, axis=1)
    if np.sum(zero_rows):
        max_indices = np.argmax(preds[zero_rows], axis=1)
        A[zero_rows, max_indices] = 1
    return A
