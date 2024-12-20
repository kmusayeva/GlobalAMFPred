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
            
            func_name = f"{method}_predict"
            
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                
                preds = func(model)

                for metric, scoring_func, kwargs in self.scores:
                    self.result.loc[metric, method] = round(scoring_func(self.Y, preds, **kwargs), 3)
            else:
                raise AttributeError(f"Method {func_name} is not defined in the class.")
        
        print(f">>>Results: \n{self.result.to_string()}")



    def ecc_predict(self, model) -> np.ndarray:
        """
        ensembles of classifer chains with random forest as base model
        """

        soft_labels = model.predict(self.X)

        preds = np.where(soft_labels > 0.5, 1, 0)    

        return preds


    def lp_predict(self, model) -> np.ndarray:
        """
        Prediction using label powerset with random forest base model.
        """

        preds = model.predict(self.X)
        
        return preds


    def mlknn_predict(self, model) -> np.ndarray:
        """
        Prediction using ml-knn.
        @return predictions
        """

        preds = model.predict(self.X)

        return preds


    def hf_predict(self, model) -> np.ndarray:
        """
        Predict with harmonic function.
        """
        
        #model = load(os.path.join(global_vars['model_dir'], "hf.pkl"))
        
        X = np.concatenate((model.X, self.X))

        dist_matrix_squared = dist_matrix(X) ** 2 ### training/validation data

        u = len(X)-len(self.X)

        train_indices = range(u)

        test_indices = range(u, len(X))

        #print(dist_matrix_squared[test_indices])

        soft_labels = model.predict(dist_matrix_squared, train_indices, test_indices)

        preds = cmn(soft_labels, model.Y_train)    

        return preds



    def cm_predict(self) -> np.ndarray:
        """
        Prediction using consistency method.
        @return predictions
        """

        X_train_valid_dist_squared = dist_matrix(self.X_train_valid) ** 2

        def objective(trial):
            
            sigma = trial.suggest_float("sigma", 0.3, 2.0, log=True)  
            
            reg = trial.suggest_float("reg", 0.1, 0.99, log=True) 
            
            soft_labels = cm(X_train_valid_dist_squared, self.Y_train_valid, self.train_idx, sigma=sigma, reg=reg)
            
            preds = cmn(soft_labels, self.Y_train)   

            f1 = f1_score(self.Y_valid, preds, average="micro")

            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)
        
        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        reg = best_trial.params["reg"]
        sigma = best_trial.params["sigma"]

        ### evaluation
        X_test_dist_squared = dist_matrix(self.X) ** 2 
        soft_labels = cm(X_test_dist_squared, self.Y, self.train_valid_idx, sigma=sigma, reg=reg)

        preds = cmn(soft_labels, self.Y_train_valid) 

        for metric, scoring_func, kwargs in self.scores:
            print(round(scoring_func(self.Y_test, preds, **kwargs), 3))


        return preds



    def knn_predict(self, models) -> np.ndarray:

        preds = np.zeros((len(self.X), self.num_species))

        for label in range(len(models)):
            preds[:, label] = models[label].predict(self.X)

        return preds



    def rf_predict(self, models) -> np.ndarray:
        """
        Predict using random forest.
        @return predictions
        """

        preds = np.zeros((len(self.X), self.num_species))

        for label in range(len(models)):
            preds[:, label] = models[label].predict(self.X)

        return preds


    def gb_predict(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Predict using gradient boosting.
        @param train_indices
        @param test_indices
        @return predictions
        """
        Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        preds = np.zeros_like(Y_test)
        for s in range(Y_train.shape[1]):
            gb = GradientBoostingClassifier(random_state=22).fit(self.species.X[train_indices], Y_train[:, s])
            preds[:, s] = gb.predict(self.species.X[test_indices])
        return preds


    def svc_predict(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Predict using support vector machine.
        @param train_indices
        @param test_indices
        @return predictions
        """
        Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        preds = np.zeros_like(Y_test)
        for s in range(Y_train.shape[1]):
            svcm = SVC(kernel='rbf', C=1000, gamma=0.001).fit(self.species.X[train_indices], Y_train[:, s])
            preds[:, s] = svcm.predict(self.species.X[test_indices])
        return preds
