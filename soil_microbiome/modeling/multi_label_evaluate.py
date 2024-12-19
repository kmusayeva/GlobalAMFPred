"""
Performs multi-label classification based on the Species object which contains all the information
of the species of interest such as the environmental variables, and species distribution.
The methods used are: ensembles of classifier chains, label powerset, harmonic function, consistency method,
ml-knn, k-nn, gradient boosting, random forest, support vector machine.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""

from .species import *
from .multi_label_classification import *
from mlp.multi_label_propagation import *
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


class MLEvaluate():
    """
    Multilabel classification framework for species classification.
    The methods used are ensemble of classifier chains, and binary relevance learners:
    harmonic  function, k-nearest neighbours, random forest, gradient boosting, support vector machine.
    The evaluation metrics used are family of F1 metrics, hamming loss, and subset accuracy.
    """
    def __init__(self, species: Species, cv: int = 5) -> None:
        super().__init__(species, cv)
        self.scores = [
            #("F1", f1_score, {"average": "samples"}),
            ("Macro-F1", f1_score, {"average": "macro"}),
            ("Micro-F1", f1_score, {"average": "micro"}),
            ("HL", hamming_loss, {}),
            ("SA", accuracy_score, {})]

        self.score_names = [x[0] for x in self.scores]
        self.methods = ["lp", "knn", "hf", "rf", "gb", "svc", "ecc", "mlknn"]
        self.result = pd.DataFrame(index=self.score_names, columns=self.methods)
        self.stds = pd.DataFrame(index=self.score_names, columns=self.methods)
        self.k, self.proportions = 2, [0.8, 0.2]


    def evaluate(self) -> None:
        """
        Evaluate the classification model using stratified sampling method.
        """
        print(f">>>Proportions for stratified sampling: {self.proportions}")
        result_nsh = {key: pd.DataFrame(columns=self.score_names, index=range(1))
                      for key in self.result.columns}
        # kf = KFold(n_splits=self.cv, shuffle=True, random_state=29)
        stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.2, 0.8])
        train_indices, valid_indices = next(stratifier.split(self.species.X_train, self.species.Y_train))

        for fold in range(1):
            for method in self.methods:
                func_name = f"{method}_predict"
                if hasattr(self, func_name):
                    func = getattr(self, func_name)
                    preds = func(train_indices, valid_indices)
                    for metric, scoring_func, kwargs in self.scores:
                        result_nsh[method].loc[fold, metric] = round(
                            scoring_func(self.species.Y_train[valid_indices], preds, **kwargs), 3)
                else:
                    raise AttributeError(f"Method {func_name} is not defined in the class.")



        for method_name in list(self.methods):
            self.result[method_name] = result_nsh[method_name].mean()
            self.stds[method_name] = result_nsh[method_name].std()


    def printResults(self):
        print(f">>>Results: \n{self.result.to_string()}")
        #print(f"Standard deviations: \n{self.stds.to_string()}")


    def ecc_predict(self) -> np.ndarray:
        """
        Prediction using Classifier Chains with a random forest base model.
        @param train_indices
        @param test_indices
        @return predictions
        """
        """
        base_model = RandomForestClassifier(random_state=49)
        ecc = EnsembleClassifierChains(base_estimator=base_model, n_chains=2, random_state=23)
        Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        ecc.fit(self.species.X[train_indices], Y_train)
        soft_labels = ecc.predict(self.species.X[test_indices])
        preds = lco(soft_labels, Y_test)  # apply one of the thresholding strategy: lco, cmn, or basic at 0.5
        """

        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

            base_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                    )

            chain = EnsembleClassifierChains(base_estimator=base_model, n_chains=2, random_state=23)

            chain.fit(self.X_train, self.Y_train)

            soft_labels = chain.predict(self.X_valid)

            preds = cmn(soft_labels, self.Y_train)    

            f1 = f1_score(self.Y_valid, preds, average="micro")
            return f1


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50, timeout=600)

        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))


        base_model = RandomForestClassifier(**best_trial.params, random_state=42)
        chain = EnsembleClassifierChains(base_estimator=base_model, n_chains=2, random_state=23)
        chain.fit(self.X_train_valid, self.Y_train_valid)
        soft_labels = chain.predict(self.X_test)
        preds = cmn(soft_labels, self.Y_test)    

        for metric, scoring_func, kwargs in self.scores:
            print(round(scoring_func(self.Y_test, preds, **kwargs), 3))


        return 1


    def lp_predict(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Prediction using label powerset with a random forest base model.
        @param train_indices
        @param test_indices
        @return predictions
        """
        base_model = RandomForestClassifier(n_estimators=20, random_state=49)
        lp = LabelPowerset(classifier=base_model, require_dense=[False, True])
        Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        lp.fit(self.species.X[train_indices], Y_train)
        preds = lp.predict(self.species.X[test_indices])
        # print(soft_labels)
        # preds = lco(soft_labels, Y_test)
        return preds


    def mlknn_predict(self) -> np.ndarray:
        """
        Prediction using ml-knn.
        @param train_indices
        @param test_indices
        @return predictions
        """
        normalized_X = StandardScaler().fit_transform(self.X_train_valid)

        def objective(trial):
            s = trial.suggest_float("s", 0.1, 0.9, log=True)  
            nn = trial.suggest_categorical("nn", [1, 3, 5, 7, 10])
            mlknn = MLkNN(k=nn, s=s)
            mlknn.fit(normalized_X[self.train_idx], self.Y_train)
            preds = mlknn.predict(normalized_X[self.valid_idx])
            f1 = f1_score(self.Y_valid, preds, average="micro")
            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)
        
        #for metric, scoring_func, kwargs in self.scores:
        #    print(round(scoring_func(self.Y[self.test_idx], preds, **kwargs), 3))


        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
    
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        normalized_X = StandardScaler().fit_transform(self.X)

        mlknn = MLkNN(k=best_trial.params["nn"], s=best_trial.params["s"])
        mlknn.fit(normalized_X[self.train_valid_idx], self.Y_train_valid)
        preds = mlknn.predict(normalized_X[self.test_idx])

        for metric, scoring_func, kwargs in self.scores:
            print(round(scoring_func(self.Y_test, preds, **kwargs), 3))

        return 1

    def hf_predict(self) -> np.ndarray:
        """
        Prediction using harmonic function.
        @param train_indices
        @param test_indices
        @return predictions
        """
        
        X_train_valid_dist_squared = dist_matrix(self.X_train_valid) ** 2 ### training/validation data

        def objective(trial):
            sigma = trial.suggest_float("sigma", 0.1, 2.0, log=True)  
            nn = trial.suggest_categorical("nn", [5, 7, 10])
            soft_labels = hf(X_train_valid_dist_squared, self.Y_train, self.train_idx, self.valid_idx, nn=nn, sigma=sigma)
            preds = lco(soft_labels, self.Y_train)    
            f1 = f1_score(self.Y_valid, preds, average="micro")
            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=500, timeout=600)
        
        best_trial = study.best_trial

        nn = best_trial.params["nn"]
        sigma = best_trial.params["sigma"]
        print(f"nn:{nn}, sigma:{sigma}")

        ### evaluation
        X_test_dist_squared = dist_matrix(self.X) ** 2 ### all data
        soft_labels = hf(X_test_dist_squared, self.Y_train_valid, self.train_valid_idx, self.test_idx, nn=nn, sigma=sigma)
        preds = cmn(soft_labels, self.Y_train_valid) 

        for metric, scoring_func, kwargs in self.scores:
            print(round(scoring_func(self.Y_test, preds, **kwargs), 3))

        return 1



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



    def knn_predict(self) -> np.ndarray:
        """
        Predict using k-nearest neighbour.
        We choose very small number of neighbour between 1 and 3.
        @return predictions
        """
        """ Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        preds = np.zeros_like(Y_test)
        normalized_X = StandardScaler().fit_transform(self.species.X)
        for s in range(Y_train.shape[1]):
            knn = KNeighborsClassifier(n_neighbors=1).fit(normalized_X[train_indices], Y_train[:, s])
            preds[:, s] = knn.predict(normalized_X[test_indices])
        """

        normalized_X = StandardScaler().fit_transform(self.X_train_valid)

        preds = np.zeros_like(self.Y_train_valid[self.valid_idx])

        nn_grid = {"nn": list(range(1, 15))}

        def objective(trial):
            
            nn = trial.suggest_categorical("nn", nn_grid["nn"])

            for s in range(self.Y_train_valid.shape[1]):
                    knn = KNeighborsClassifier(n_neighbors=nn).fit(normalized_X[self.train_idx], self.Y_train_valid[self.train_idx, s])
                    preds[:, s] = knn.predict(normalized_X[self.valid_idx])

            f1 = f1_score(self.Y_train_valid[self.valid_idx], preds, average="micro")
            return f1


        sampler = optuna.samplers.GridSampler(nn_grid)

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=len(nn_grid["nn"]))
        
        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))


        nn = best_trial.params["nn"]
        
        normalized_X = StandardScaler().fit_transform(self.X)

        preds = np.zeros_like(self.Y[self.test_idx])

        for s in range(self.Y_train_valid.shape[1]):
                knn = KNeighborsClassifier(n_neighbors=nn).fit(normalized_X[self.train_valid_idx], self.Y[self.train_valid_idx, s])
                preds[:, s] = knn.predict(normalized_X[self.test_idx])


        for metric, scoring_func, kwargs in self.scores:
            print(round(scoring_func(self.Y[self.test_idx], preds, **kwargs), 3))


        return 1



    def rf_predict(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Predict using random forest.
        @param train_indices
        @param test_indices
        @return predictions
        """
        Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        preds = np.zeros_like(Y_test)
        for s in range(Y_train.shape[1]):
            rf = RandomForestClassifier(random_state=73).fit(self.species.X[train_indices], Y_train[:, s])
            preds[:, s] = rf.predict(self.species.X[test_indices])
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
