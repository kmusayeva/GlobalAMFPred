"""
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""

from .multi_label_classification import *
from .harmonic_function import *
from .autogluon_multilabel_predictor import *
from mlp.thresholding import *
from joblib import load
from autogluon.common.utils.log_utils import set_logger_verbosity
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

set_logger_verbosity(3)


class MLEvaluate(MLClassification):
    """
    Evaluate trained models.
    The methods used are ensembles of classifier chains, label powerset, harmonic function,
    ml-knn, k-nn, gradient boosting, random forest, support vector machine,
    xgboost, lightgbm, autogluon.
    """

    def __init__(self, species: Species, method: Optional[List[str]] = None):

        super().__init__(species, method)

        self.result = pd.DataFrame(index=self.score_names, columns=self.methods)

        self.mcc = pd.DataFrame(index=self.species_names, columns=self.methods)

        self.mcc.index.name = "species"

        self.X_column_names = species.X.columns.tolist()

        self.Y_column_names = species.Y_top.columns.tolist()

    def evaluate(self) -> None:
        """
        Evaluate the classification model using stratified sampling method.
        """

        for method in self.methods:

            if method == "autogluon":

                print(f">>>Evaluating model: autogluon...")

                preds = self.autogluon_predict()
                print(preds)
                if preds is None: continue

            else:

                file_name = os.path.join(global_vars["model_dir"], f"{method}.pkl")

                try:
                    model = load(file_name)
                except FileNotFoundError:
                    print(f"Error: The file '{file_name}' does not exist.")
                    continue
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue

                print(f">>>Evaluating model: {self.methods_long[method]}...")

                if method in ["ecc", "lp", "xgb", "mlknn"]:
                    preds = self.multi_label_predict(model)
                    if method == "ecc":
                        preds = np.rint(preds)

                elif method in ["knn", "rf", "gb", "svc", "lgbm"]:
                    preds = self.binary_relevance_predict(model)

                elif method == "hf":
                    preds = self.hf_predict(model)

            preds = preds[:, : self.num_species]

            for metric, scoring_func, kwargs in self.scores:
                self.result.loc[metric, method] = round(
                    scoring_func(self.Y, preds, **kwargs), 3
                    )

            if method=="lp": preds = preds.toarray()
            for i in range(self.num_species):
                self.mcc.loc[self.species_names[i], method] = round(
                    matthews_corrcoef(self.Y[:,i], preds[:,i]), 3
                )

        print(f">>>Results: \n{self.result.to_string()}")
        print(f"\nMatthews correlation coefficient:\n {self.mcc.to_string()}")

        print(">>>Confusion matrices for the most frequent and the rarest species:\n")
        print(f"{self.species_names[0]}")
        print(confusion_matrix(self.Y[:, 0], preds[:,0]))
        print(f"{self.species_names[19]}")
        print(confusion_matrix(self.Y[:, 19], preds[:, 19]))


    def multi_label_predict(self, model) -> np.ndarray:
        """
        Predict with ecc, lp, xgb.
        """

        # Predict simultaneously for all labels
        preds = model.predict(self.X)

        return preds

    def binary_relevance_predict(self, models) -> np.ndarray:
        """
        Predict with knn, rf, gb, svc using binary relevance.
        """

        preds = np.zeros((len(self.X), self.num_species))

        # Predict for each label separately
        for label in range(self.num_species):
            preds[:, label] = models[label].predict(self.X)

        return preds

    def hf_predict(self, model) -> np.ndarray:
        """
        Predict with harmonic function.
        """

        # Concatenate training and test data
        X = np.concatenate((model.X, self.X))

        # Compute the squared distance matrix
        dist_matrix_squared = dist_matrix(X) ** 2

        # Get the indices for training and test data
        u = len(X) - len(self.X)

        train_indices = range(u)

        test_indices = range(u, len(X))

        soft_labels = model.predict(dist_matrix_squared, train_indices, test_indices)

        preds = basic(soft_labels, model.Y_train) # threshold at 0.5

        return preds

    def autogluon_predict(self):
        """
        Prediction using autogluon for each label separately.
        """

        try:
            root_model_dir = os.path.join(global_vars["model_dir"], "autogluon")

            if not os.path.exists(root_model_dir):
                raise FileNotFoundError(f"Model directory '{root_model_dir}' does not exist.")

            test_data = pd.DataFrame(self.X, columns=self.X_column_names).join(
                pd.DataFrame(self.Y, columns=self.Y_column_names)
                )

            all_predictions = pd.DataFrame()

            try:
                label_dirs = os.listdir(root_model_dir)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not list contents of '{root_model_dir}'.")

            for label_dir in label_dirs:
                if (label_dir == "multilabel_predictor.pkl") or not (
                        "_".join(label_dir.split("_")[1:]) in self.species_names
                ):
                    continue
                label_model_path = os.path.join(root_model_dir, label_dir)

                try:
                    predictor = TabularPredictor.load(label_model_path, verbosity=0)
                except FileNotFoundError:
                    print(f"Warning: Model file '{label_model_path}' not found. Skipping.")
                    continue
                except Exception as e:
                    print(f"Error loading model from '{label_model_path}': {e}")
                    continue

                try:
                    label_predictions = predictor.predict(test_data)
                except Exception as e:
                    print(f"Error predicting with model '{label_dir}': {e}")
                    continue

                all_predictions[label_dir] = label_predictions

            new_names = [
                "_".join(name.split("_")[1:]) for name in all_predictions.columns.tolist()
            ]

            all_predictions.columns = new_names

            preds = all_predictions[self.Y_column_names].to_numpy()

            return preds

        except Exception as e:
            print(f"An error occurred in the function: {e}")
            return None

