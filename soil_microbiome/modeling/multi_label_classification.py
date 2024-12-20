"""
Performs multi-label classification based on the Species object which contains all the information
of the species of interest such as the environmental variables, and species distribution.
The methods used are: ensembles of classifier chains, label powerset, harmonic function, consistency method,
ml-knn, k-nn, gradient boosting, random forest, support vector machine.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""

from .species import *
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, hamming_loss, make_scorer


class MLClassification:
    
    """
    Multilabel classification framework for species classification.
    The methods used are ensemble of classifier chains, and binary relevance learners:
    harmonic  function, k-nearest neighbours, random forest, gradient boosting, support vector machine.
    The evaluation metrics used are family of F1 metrics, hamming loss, and subset accuracy.
    """

    def __init__(self, species: Species, cv: int = 5) -> None:
        
        self.cv = cv

        self.X = species.X.to_numpy()
        
        self.Y = species.Y_top.to_numpy()

        self.species_names = species.Y_top.columns.tolist()

        self.num_species = species.num_species_interest

        # define evaluation metrics
        self.scores = [
            #("F1", f1_score, {"average": "samples"}),
            ("Macro-F1", f1_score, {"average": "macro"}),
            ("Micro-F1", f1_score, {"average": "micro"}),
            ("SA", accuracy_score, {}),                   
            ("HL", hamming_loss, {})
            ]

        self.score_names = [x[0] for x in self.scores]

        ### specify which models are used

        #self.methods = ["lp", "knn", "hf", "rf", "gb", "svc", "ecc", "mlknn"]
        
        self.methods = ["knn", "mlknn", "hf", "rf", "ecc", "lp"]