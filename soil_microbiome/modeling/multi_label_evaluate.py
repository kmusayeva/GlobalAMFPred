"""
Performs multi-label classification based on the Species object which contains all the information
of the species of interest such as the environmental variables, and species distribution.
The methods used are: ensembles of classifier chains, label powerset, harmonic function, consistency method,
ml-knn, k-nn, gradient boosting, random forest, support vector machine.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""

from .multi_label_classification import *
from mlp.multi_label_propagation import *
from .harmonic_function import *
from mlp.thresholding import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import ClassifierChain
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
import pickle
from joblib import load


class MLEvaluate(MLClassification):
    """
    Multilabel classification framework for species classification.
    The methods used are ensemble of classifier chains, and binary relevance learners:
    harmonic  function, k-nearest neighbours, random forest, gradient boosting, support vector machine.
    The evaluation metrics used are family of F1 metrics, hamming loss, and subset accuracy.
    """
    def __init__(self, species: Species) -> None:
        
        super().__init__(species)

        self.result = pd.DataFrame(index=self.score_names, columns=self.methods)


    def evaluate(self) -> None:
        """
        Evaluate the classification model using stratified sampling method.
        """

        for method in self.methods:
            
            file_name = os.path.join(global_vars['model_dir'], f"{method}.pkl")

            model = load(file_name)

            print(f">>>Evaluating model: {self.methods_long[method]}...")

            if method in ["ecc", "lp", "xgb", "mlknn"]:
                preds = self.multi_label_predict(model)
                if method == "ecc": preds = np.rint(preds)
                
            elif method in ["knn", "rf", "gb", "svc"]:
                preds = self.binary_relevance_predict(model)

            elif method == "hf":
                preds = self.hf_predict(model)


            preds = preds[:, :self.num_species]

            for metric, scoring_func, kwargs in self.scores:
                self.result.loc[metric, method] = round(scoring_func(self.Y, preds, **kwargs), 3)

        
        print(f">>>Results: \n{self.result.to_string()}")



    def multi_label_predict(self, model) -> np.ndarray:
        """
        predict with ecc, lp, xgb
        """

        preds = model.predict(self.X)

        return preds


    def binary_relevance_predict(self, models) -> np.ndarray:
        """
        predict with knn, rf, gb, svc
        """
       
        preds = np.zeros((len(self.X), self.num_species))

        for label in range(self.num_species):
            preds[:, label] = models[label].predict(self.X)

        return preds


    def hf_predict(self, model) -> np.ndarray:
        """
        Predict with harmonic function.
        """
        
        X = np.concatenate((model.X, self.X))

        dist_matrix_squared = dist_matrix(X) ** 2 ### training/validation data

        u = len(X)-len(self.X)

        train_indices = range(u)

        test_indices = range(u, len(X))

        #print(dist_matrix_squared[test_indices])

        soft_labels = model.predict(dist_matrix_squared, train_indices, test_indices)

        preds = basic(soft_labels, model.Y_train)    

        return preds


    def cm_predict(self) -> np.ndarray:
        """
        Prediction using consistency method.
        @return predictions
        """

        ### evaluation
        X_test_dist_squared = dist_matrix(self.X) ** 2 
        soft_labels = cm(X_test_dist_squared, self.Y, self.train_valid_idx, sigma=sigma, reg=reg)

        preds = cmn(soft_labels, self.Y_train_valid) 

        for metric, scoring_func, kwargs in self.scores:
            print(round(scoring_func(self.Y_test, preds, **kwargs), 3))


        return preds

