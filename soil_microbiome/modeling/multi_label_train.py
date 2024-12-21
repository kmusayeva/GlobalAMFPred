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
import xgboost as xgb
from skmultilearn.problem_transform import LabelPowerset
import pickle
import optuna
#optuna.logging.set_verbosity(optuna.logging.WARNING)


class MLTrain(MLClassification):

    def __init__(self, species: Species, cv: int = 5) -> None:
    
        super().__init__(species, cv)

        # split data into train/test sets

        stratifier = IterativeStratification(n_splits=2, order=5, sample_distribution_per_fold=[0.2, 0.8])

        self.train_idx, self.test_idx = next(stratifier.split(self.X, self.Y))

        self.X_train, self.X_test = self.X[self.train_idx], self.X[self.test_idx]

        self.Y_train, self.Y_test = self.Y[self.train_idx], self.Y[self.test_idx]


    def train(self) -> None:
        """
        Model training.
        """

        for method in self.methods:

            func_name = f"{method}_train"

            print("Training model: ", method)

            if hasattr(self, func_name):

                func = getattr(self, func_name)

                model = func()

                print(f">>>>Done: {func_name}")

                file_name = os.path.join(global_vars['model_dir'], f"{method}.pkl")

                print(f"Saving {func_name} to a file.")

                with open(file_name, "wb") as file:
                    pickle.dump(model, file)

                print(f"Model saved to {file_name}")

            else:
                raise AttributeError(f"Method {func_name} is not defined in the class.")



    def ecc_train(self):
        """
        Ensembles of classifier chains with random forest as base model.
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

            preds = np.rint(soft_labels)

            f1 = f1_score(self.Y_test, preds, average="micro")
            return f1


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)

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
        
        X_dist_squared = dist_matrix(self.X) ** 2 ### training/validation data

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



    def lp_train(self):
        """
        Train label powerset with random forest base model.
        """

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20)
                }


            base_model = RandomForestClassifier(**params, random_state=42)

            model = LabelPowerset(classifier=base_model, require_dense=[False, True])

            model.fit(self.X_train, self.Y_train)

            preds = model.predict(self.X_test)
            #print(soft_labels)
            #preds = lco(soft_labels, self.Y_train)    

            f1 = f1_score(self.Y_test, preds, average="micro")

            return f1


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)

        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))


        base_model = RandomForestClassifier(**best_trial.params, random_state=42)
        
        model = LabelPowerset(classifier=base_model, require_dense=[False, True])

        model.fit(self.X, self.Y)

        return model


    def mlknn_train(self):
        """
        Prediction using ml-knn.
        """

        def objective(trial):

            s = trial.suggest_float("s", 0.1, 0.9, log=True) 

            nn = trial.suggest_categorical("nn", [1, 3, 5, 7, 10])

            model = MLkNN(k=nn, s=s)
            
            model.fit(self.X_train, self.Y_train)

            preds = model.predict(self.X_test)

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

        for s in range(self.Y.shape[1]):
                
                model = KNeighborsClassifier(n_neighbors=nn).fit(self.X, self.Y[:, s])

                models[s] = model

        return models


    
    def rf_train(self):
        """
        Random forest.
        """
       
        preds = np.zeros_like(self.Y_test)

        def objective(trial):

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20)
                }
            
            for label in range(self.Y_train.shape[1]):
                rf = RandomForestClassifier(**params, random_state=42)
                model = rf.fit(self.X_train, self.Y_train[:, label])
                preds[:, label] = model.predict(self.X_test)

            f1 = f1_score(self.Y_test, preds, average="micro")
            return f1


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)

        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))


        models = {}

        for label in range(self.Y.shape[1]):
            rf = RandomForestClassifier(**best_trial.params, random_state=42)
            models[label] = rf.fit(self.X, self.Y[:, label])


        return models


    def gb_train(self):
        """
        Gradient boosting.
        """

        preds = np.zeros_like(self.Y_test)

        def objective(trial):

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0)
            }    

            for label in range(self.Y_train.shape[1]):
                gb = GradientBoostingClassifier(**params, random_state=42)
                model = gb.fit(self.X_train, self.Y_train[:, label])
                preds[:, label] = model.predict(self.X_test)

            f1 = f1_score(self.Y_test, preds, average="micro")
            return f1


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        best_trial = study.best_trial
        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))


        models = {}

        for label in range(self.Y.shape[1]):
            gb = GradientBoostingClassifier(**best_trial.params, random_state=42)
            models[label] = gb.fit(self.X, self.Y[:, label])

        return models           



    def xgb_train(self):
        dtrain = xgb.DMatrix(self.X_train, label=self.Y_train)
        dvalid = xgb.DMatrix(self.X_test, label=self.Y_test)

        def objective_fungi(trial):

            param = {
                "verbosity": 0,
                "objective": "binary:logistic",
                # use exact for small dataset.
                "tree_method": "hist", ### do not forget to change it from exact to hist if you use categorical variables
                # defines booster, gblinear for linear functions.
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }

            if param["booster"] in ["gbtree", "dart"]:
                param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
                param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
                param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
                param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)


            model = xgb.train(param, dtrain)
            soft_labels = model.predict(dvalid)
            preds = np.rint(soft_labels)
            f1 = f1_score(self.Y_test, preds, average="macro")
            return f1


        study = optuna.create_study(direction="maximize")
        study.optimize(objective_fungi, n_trials=100, timeout=600)
                
        best_trial = study.best_trial
        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        model = xgb.XGBClassifier(**best_trial.params)
        model.fit(self.X, self.Y)

        return model


    def svc_train(self):
        """
        Support vector machine.
        """

        preds = np.zeros_like(self.Y_test)
        
        def objective(trial):

            params = {
                "C": trial.suggest_float("C", 1e-3, 1e2),
                "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True) 
                }

            for label in range(self.Y_train.shape[1]):
                svcm = SVC(**params, kernel="rbf", random_state=42).fit(self.X_train, self.Y_train[:, label])
                preds[:, label] = svcm.predict(self.X_test)


            f1 = f1_score(self.Y_test, preds, average="micro")
            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)

        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))

        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))


        models = {}

        for label in range(self.Y.shape[1]):
            svcm = SVC(**best_trial.params, kernel="rbf", random_state=42).fit(self.X, self.Y[:, label])
            models[label] = svcm


        return models




    def cm_train(self):
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

