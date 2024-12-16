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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, hamming_loss, make_scorer
from .species import *
from .species_europe import *


class SpeciesClassification:

    def __init__(self, species, cv=5):
        self.species = species
        self.cv = cv
        self.species.X = self.species.X.to_numpy()
        self.species.Y = self.species.Y_top.to_numpy()
        self.species_names = species.Y_top.columns.tolist()
