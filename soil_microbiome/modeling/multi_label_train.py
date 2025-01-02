"""
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""

from .multi_label_classification import *
from .harmonic_function import *
from .ensembles_cchains import *
from .autogluon_multilabel_predictor import *
from mlp.multi_label_propagation import *
from mlp.thresholding import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from skmultilearn.adapt import MLkNN
import xgboost as xgb
from skmultilearn.problem_transform import LabelPowerset
import pickle
import optuna

# optuna.logging.set_verbosity(optuna.logging.WARNING)


class MLTrain(MLClassification):
    """
    Perform multi-label classification based on Species object
    which contains the environmental (input) and species (output) data.
    The methods used are ensembles of classifier chains, label powerset, harmonic function,
    k-nn, ml-knn, support vector machine, random forest, gradient boosting, xgboost, lightgbm, autogluon.
    """

    def __init__(self, species: Species, method: Optional[List[str]] = None):

        super().__init__(species, method)

        # split data into train/test sets using stratified sampling
        stratifier = IterativeStratification(
            n_splits=2, order=5, sample_distribution_per_fold=[0.2, 0.8]
        )

        self.train_idx, self.test_idx = next(stratifier.split(self.X, self.Y))

        self.X_train, self.X_test = self.X[self.train_idx], self.X[self.test_idx]

        self.Y_train, self.Y_test = self.Y[self.train_idx], self.Y[self.test_idx]

        self.X_column_names = species.X.columns

        self.Y_column_names = species.Y_top.columns

    def train(self) -> None:
        """
        Train all models specified in the parent class.
        """

        for method in self.methods:

            if method == "autogluon":  # treat autogluon separately
                print("Training the model: autogluon")
                self.autogluon_train()

            else:
                func_name = f"{method}_train"

                print("Training the model: ", self.methods_long[method])

                if hasattr(self, func_name):

                    func = getattr(self, func_name)

                    model = func()

                    print(f">>>>Done: {self.methods_long[method]}")

                    file_name = os.path.join(global_vars["model_dir"], f"{method}.pkl")

                    print(f"Saving {func_name} to a file.")

                    with open(file_name, "wb") as file:
                        pickle.dump(model, file)

                    print(f"Model saved to {file_name}")

                else:
                    raise AttributeError(
                        f"Method {func_name} is not defined in the class."
                    )

    def ecc_train(self):
        """
        Train ensembles of classifier chains with random forest as base model.
        """

        def objective(trial):
            # hyperparams of random forest
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }

            base_model = RandomForestClassifier(**params, random_state=42)

            model = EnsembleClassifierChains(
                base_estimator=base_model, n_chains=2, random_state=23
            )

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

        model = EnsembleClassifierChains(
            base_estimator=base_model, n_chains=2, random_state=23
        )

        model.fit(self.X_train, self.Y_train)

        return model

    def lp_train(self):
        """
        Train label powerset with random forest as base model.
        """

        def objective(trial):
            """
            # hyperparams of svc
            params = {
                "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "sigmoid"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            }

            # Create the base model using the suggested parameters
            base_model = SVC(**params, random_state=42)
            """
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }

            base_model = RandomForestClassifier(**params, random_state=42)

            # Wrap the base model with Label Powerset
            model = LabelPowerset(classifier=base_model, require_dense=[False, True])

            # Fit the model on the training data
            model.fit(self.X_train, self.Y_train)

            # Predict on the test data
            preds = model.predict(self.X_test)

            # Calculate the F1 score (micro-averaged)
            f1 = f1_score(self.Y_test, preds, average="micro")

            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=150, timeout=600)

        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        base_model = RandomForestClassifier(**best_trial.params, random_state=42)

        model = LabelPowerset(classifier=base_model, require_dense=[False, True])

        model.fit(self.X, self.Y)

        return model

    def hf_train(self):
        """
        Train harmonic function.
        """

        X_dist_squared = (
            dist_matrix(self.X) ** 2
        )  # distance matrix of training and validation data

        def objective(trial):

            sigma = trial.suggest_float("sigma", 0.1, 2.0, log=True)

            nn = trial.suggest_categorical("nn", [5, 7, 10])

            model = HarmonicFunction(sigma=sigma, nn=nn)

            model.fit(self.X, self.Y_train)

            soft_labels = model.predict(X_dist_squared, self.train_idx, self.test_idx)

            preds = cmn(soft_labels, self.Y_train)

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

    def mlknn_train(self):
        """
        Train multi-label knn.
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

        # normalized_X = StandardScaler().fit_transform(self.X_train)

        preds = np.zeros_like(self.Y_test)

        # set the range for the number of nearest neighbours
        nn_grid = {"nn": list(range(1, 15))}

        def objective(trial):

            nn = trial.suggest_categorical("nn", nn_grid["nn"])

            for s in range(self.Y_train.shape[1]):
                knn = KNeighborsClassifier(n_neighbors=nn).fit(
                    self.X_train, self.Y_train[:, s]
                )
                preds[:, s] = knn.predict(self.X_test)

            f1 = f1_score(self.Y_test, preds, average="micro")

            return f1

        # only one hyperparam, so do grid search
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
        Train random forest.
        """

        preds = np.zeros_like(self.Y_test)

        def objective(trial):
            # hyperparams
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
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

    def lgbm_train(self):
        """
        Train LightGBM.
        """

        def objective(trial):
            # hyperparams
            param = {
                "objective": "binary",
                "boosting_type": "gbdt",
                "metric": "None",
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0, step=0.1
                ),
            }

            models = []

            for i in range(self.Y_train.shape[1]):
                clf = LGBMClassifier(**param, random_state=42, verbose=-1)
                clf.fit(self.X_train, self.Y_train[:, i])
                models.append(clf)

            preds = np.zeros(self.Y_test.shape)
            for i, model in enumerate(models):
                preds[:, i] = model.predict(self.X_test)

            f1_micro = f1_score(self.Y_test, preds, average="micro")
            return f1_micro

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)

        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        models = {}

        for label in range(self.Y.shape[1]):
            lgbm = LGBMClassifier(**best_trial.params, random_state=42, verbose=-1)
            models[label] = lgbm.fit(self.X, self.Y[:, label])

        return models

    def gb_train(self):
        """
        Train gradient boosting.
        """

        preds = np.zeros_like(self.Y_test)

        def objective(trial):
            # hyperparams
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 1.0, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }

            for label in range(self.Y_train.shape[1]):
                gb = GradientBoostingClassifier(**params, random_state=42)
                model = gb.fit(self.X_train, self.Y_train[:, label])
                preds[:, label] = model.predict(self.X_test)

            f1 = f1_score(self.Y_test, preds, average="micro")
            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

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
        """
        Train extreme gradient boosting.
        """

        dtrain = xgb.DMatrix(self.X_train, label=self.Y_train)
        dvalid = xgb.DMatrix(self.X_test, label=self.Y_test)

        def objective_fungi(trial):
            # hyperparams
            param = {
                "verbosity": 0,
                "objective": "binary:logistic",
                "tree_method": "hist",
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }

            if param["booster"] in ["gbtree", "dart"]:
                param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
                param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
                param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                param["grow_policy"] = trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                )

            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical(
                    "sample_type", ["uniform", "weighted"]
                )
                param["normalize_type"] = trial.suggest_categorical(
                    "normalize_type", ["tree", "forest"]
                )
                param["rate_drop"] = trial.suggest_float(
                    "rate_drop", 1e-8, 1.0, log=True
                )
                param["skip_drop"] = trial.suggest_float(
                    "skip_drop", 1e-8, 1.0, log=True
                )

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
        Train support vector machine.
        """

        preds = np.zeros_like(self.Y_test)

        def objective(trial):
            # hyperparams
            params = {
                "C": trial.suggest_float("C", 1e-3, 1e2),
                "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
            }

            for label in range(self.Y_train.shape[1]):
                svcm = SVC(**params, kernel="rbf", random_state=42).fit(
                    self.X_train, self.Y_train[:, label]
                )
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
            svcm = SVC(**best_trial.params, kernel="rbf", random_state=42).fit(
                self.X, self.Y[:, label]
            )
            models[label] = svcm

        return models

    def autogluon_train(self):
        """
        Train autogluon multi-label classifier.
        Trains for each label separately.
        """
        df_train = pd.DataFrame(self.X_train, columns=self.X_column_names).join(
            pd.DataFrame(self.Y_train, columns=self.Y_column_names)
        )
        df_test = pd.DataFrame(self.X_test, columns=self.X_column_names).join(
            pd.DataFrame(self.Y_test, columns=self.Y_column_names)
        )

        labels = self.species_names
        problem_types = ["binary"] * len(labels)
        eval_metrics = ["f1"] * len(labels)
        path = os.path.join(global_vars["model_dir"], "autogluon/")

        model = MultilabelPredictor(
            labels=labels,
            problem_types=problem_types,
            eval_metrics=eval_metrics,
            path=path,
        )

        model.fit(df_train, tuning_data=df_test)

        return model
