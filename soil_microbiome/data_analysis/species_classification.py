"""
Base class for species classification.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.multioutput import ClassifierChain
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, hamming_loss, make_scorer
from .species import *


class SpeciesClassification:

    def __init__(self, species, cv=5):
        
        self.cv = cv

        self.X = species.X.to_numpy()
        
        self.Y = species.Y_top.to_numpy()

        stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.2, 0.8])

        self.train_valid_idx, self.test_idx = next(stratifier.split(species.X, species.Y))

        self.X_train_valid, self.Y_train_valid = self.X[self.train_valid_idx], self.Y[self.train_valid_idx]

        stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.25, 0.75])

        self.train_idx, self.valid_idx = next(stratifier.split(self.X_train_valid, self.Y_train_valid))

        self.species_names = species.Y_top.columns.tolist()


