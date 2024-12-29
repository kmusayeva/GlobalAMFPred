"""
Implements different thresholding strategies.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
import numpy as np
thresholds = [x/100.0 for x in range(1, 90, 1)]

def lco(preds, Y_train):
    """
    Implements the method of Read et al. 2011
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



def cmn(preds, Y_train):
    """
    Implements class-mass normalization method of Zhu et al., 2003
    """
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


def basic(preds, Y_train=None):
    """
    Implements thresholding at 0.5
    """
    A = np.where(preds >= 0.5, 1, 0)
    A = correct_all_zero_rows(A, preds)
    return A

def correct_all_zero_rows(A, preds):
    """
    If prediction is zero for all labels, then set to 1 the highest soft-label value
    """
    zero_rows = np.all(A == 0, axis=1)
    if np.sum(zero_rows):
        max_indices = np.argmax(preds[zero_rows], axis=1)
        A[zero_rows, max_indices] = 1
    return A
