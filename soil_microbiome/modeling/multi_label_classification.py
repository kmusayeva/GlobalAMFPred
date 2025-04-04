"""
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""

from .species import *
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    make_scorer,
)


class MLClassification:
    """
    Multilabel classification framework for species classification.
    The methods used are ensembles of classifier chains, label powerset, harmonic function, consistency method,
    ml-knn, k-nn, gradient boosting, random forest, support vector machine, xgboost, lightgbm, autogluon.
    The evaluation metrics used are family of F1 metrics, hamming loss, and subset accuracy.
    """

    def __init__(self, species: Species, method: Optional[List[str]] = None):

        self.X = species.X.to_numpy()

        self.Y = species.Y_top.to_numpy()

        self.species_names = species.Y_top.columns.tolist()

        self.num_species = species.num_species_interest

        # define evaluation metrics
        self.scores = [
            # ("F1", f1_score, {"average": "samples"}),
            ("Macro-F1", f1_score, {"average": "macro"}),
            ("Micro-F1", f1_score, {"average": "micro"}),
            ("SA", accuracy_score, {}),
            ("HL", hamming_loss, {}),
        ]

        self.score_names = [x[0] for x in self.scores]

        # learning methods used
        self.methods_long = {
            "knn": "k-nearest neighbors",
            "mlknn": "multi-label knn",
            "hf": "harmonic function",
            "svc": "support vector machine",
            "rf": "random forest",
            "gb": "gradient boosting",
            "xgb": "extreme gradient boosting",
            "ecc": "ensemble of classifier chains",
            "lp": "label powerset",
            "autogluon": "autogluon",
            "lgbm": "lightgbm",
        }

        self.methods = [
            "knn",
            "hf",
            "svc",
            "rf",
            "gb",
            "xgb",
            "ecc",
            "lp",
            "lgbm",
            "autogluon",
        ]

        if method is not None:

            if (len(method) == 1) and (len(method[0].split("-")) > 1):
                self.methods = [m for m in self.methods if m != method[0].split("-")[1]]

            elif set(method).issubset(self.methods):
                self.methods = method
