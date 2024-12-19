"""
Implements ensemble of classifier chains of Read et al.  2011
Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.svm import SVC


class EnsembleClassifierChains:

    def __init__(self, base_estimator, n_chains: int = 5,
                 random_state: int = None) -> None:
        """
        @param base_estimator: random forest or support vector machine
        @param n_chains: size of the ensemble of classifier  chains
        @param random_state: randomness of the chain order
        """
        self.base_estimator = base_estimator
        self.n_chains = n_chains
        self.random_state = random_state
        self.chains = []
        self.label_orderings = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        @param X: input data
        @param Y: multi-label output data
        """
        rng = np.random.RandomState(self.random_state)
        n_labels = Y.shape[1]
        for i in range(self.n_chains):
            # Randomly permute the order of labels for this chain.
            label_order = list(rng.permutation(n_labels))
            self.label_orderings.append(label_order)

            # Initialize ClassifierChain with the base estimator and the current label order.
            chain = ClassifierChain(self.base_estimator, order=label_order, random_state=rng)
            chain.fit(X, Y)
            self.chains.append(chain)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        We take the mean value of classifier predictions in the ensemble.
        The obtained soft labels is then subject to a thresholding.
        @param X: input data
        @return: soft labels
        """
        Y_pred_chains = np.array([chain.predict_proba(X) for chain in self.chains])
        Y_pred = np.mean(Y_pred_chains, axis=0)
        return Y_pred
