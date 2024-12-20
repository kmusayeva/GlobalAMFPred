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
from .harmonic_function import *
from .ensembles_cchains import *
from mlp.multi_label_propagation import *
from mlp.thresholding import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import ClassifierChain
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import LabelPowerset
import pickle
import optuna
#optuna.logging.set_verbosity(optuna.logging.WARNING)


class MLTrain(MLClassification):

    def __init__(self, species: Species, cv: int = 5) -> None:
    
        super().__init__(species, cv)
    

    def train(self) -> None:
        """
        Train the models.
        """

        for method in self.methods:

            func_name = f"{method}_train"

            if hasattr(self, func_name):

                func = getattr(self, func_name)

                model = func()

                print(f">>>>Done: {func_name}")

                file_name = os.path.join(global_vars['model_dir'], f"{method}.pkl")

                print(f"Saving {func_name} to a file.")

                with open(file_name, "wb") as file:tonality
                    pickle.dump(model, file)

                print(f"Model saved to {file_name}")

            else:
                raise AttributeError(f"Method {func_name} is not defined in the class.")



    def ecc_train(self):

        """
        Train ensembles of classifier chains with random forest as base model.
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

            model = EnsembleClassifierChains(base_estimator=base_model, n_chains=2, random_state=23)

            model.fit(self.X_train, self.Y_train)

            soft_labels = model.predict(self.X_test)

            preds = basic(soft_labels, self.Y_train)    

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
        
        model = EnsembleClassifierChains(base_estimator=base_model, n_chains=2, random_state=23)
        
        model.fit(self.X_train, self.Y_train)

        return model


    def hf_train(self):
        """
        Train harmonic function.
        """
        
        X_dist_squared = dist_matrix(self.X_train) ** 2 ### training/validation data

        def objective(trial):
            
            sigma = trial.suggest_float("sigma", 0.1, 2.0, log=True)  
            
            nn = trial.suggest_categorical("nn", [5, 7, 10])
            
            model = HarmonicFunction(sigma=sigma, nn=nn)
            
            model.fit(self.X, self.Y_train)
            
            soft_labels = model.predict(X_dist_squared, self.train_idx, self.test_idx)
            
            preds = basic(soft_labels, self.Y_train)    

            f1 = f1_score(self.Y_test, preds, average="micro")

            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)
        
        best_trial = study.best_trial

        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        nn = best_trial.params["nn"]
        
        sigma = best_trial.params["sigma"]

        model = HarmonicFunction(sigma=sigma, nn=nn)
        
        model.fit(self.X, self.Y)

        return model



    def lp_train(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
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


    def mlknn_train(self) -> np.ndarray:
        """
        Prediction using ml-knn.
        @param train_indices
        @param test_indices
        @return predictions
        """

        def objective(trial):
            s = trial.suggest_float("s", 0.1, 0.9, log=True) 

            nn = trial.suggest_categorical("nn", [1, 3, 5, 7, 10])

            model = MLkNN(k=nn, s=s)
            
            model.fit(self.X_train, self.Y_train)

            preds = mlknn.predict(self.X_test)

            f1 = f1_score(self.Y_test, preds, average="micro")
            
            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)
        
        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
    
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))


        model = MLkNN(k=best_trial.params["nn"], s=best_trial.params["s"])

        model.fit(self.X, self.Y)

        return model



    def cm_train(self) -> np.ndarray:
        """
        Prediction using consistency method.
        @return predictions
        """

        X_dist_squared = dist_matrix(self.X) ** 2

        def objective(trial):
            
            sigma = trial.suggest_float("sigma", 0.3, 2.0, log=True)  
            
            reg = trial.suggest_float("reg", 0.1, 0.99, log=True) 
            
            soft_labels = cm(X_dist_squared, self.Y, self.train_idx, sigma=sigma, reg=reg)
            
            preds = cmn(soft_labels, self.Y_train)   

            f1 = f1_score(self.Y_test, preds, average="micro")

            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)
        
        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        reg = best_trial.params["reg"]
        sigma = best_trial.params["sigma"]

        ### update

        return 1



    def knn_train(self):
        """
        Train k-nearest neighbour.
        """

        normalized_X = StandardScaler().fit_transform(self.X_train)

        preds = np.zeros_like(self.Y_test)

        # set the range for the number of nearest neighbours
        nn_grid = {"nn": list(range(1, 15))}

        def objective(trial):
            
            nn = trial.suggest_categorical("nn", nn_grid["nn"])

            for s in range(self.Y_train.shape[1]):
                    knn = KNeighborsClassifier(n_neighbors=nn).fit(self.X_train, self.Y_train[:, s])
                    preds[:, s] = knn.predict(self.X_test)

            f1 = f1_score(self.Y_test, preds, average="micro")

            return f1


        sampler = optuna.samplers.GridSampler(nn_grid)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=len(nn_grid["nn"]))
        
        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        nn = best_trial.params["nn"]

        models = {}

        for s in range(self.Y_train_valid.shape[1]):
                
                model = KNeighborsClassifier(n_neighbors=nn).fit(self.X, self.Y[:, s])

                models[s] = model

        return models



    def rf_train(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
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


    def gb_train(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
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


    def svc_train(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
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
