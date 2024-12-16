"""
Performs multi-label classification based on the Species object which contains all the information
of the species of interest such as the environmental variables, and species distribution.
The methods used are: ensembles of classifier chains, label powerset, harmonic function, consistency method,
ml-knn, k-nn, gradient boosting, random forest, support vector machine.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
from sklearn.utils import shuffle
from .species_classification import *
from .stratified_sampling import *
from .ensembles_cchains import *

from mlp.multi_label_propagation import *
from mlp.thresholding import *
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import LabelPowerset


class MLClassification(SpeciesClassification):
    """
    Multilabel classification framework for species classification.
    The methods used are ensemble of classifier chains, and binary relevance learners:
    harmonic  function, k-nearest neighbours, random forest, gradient boosting, support vector machine.
    The evaluation metrics used are family of F1 metrics, hamming loss, and subset accuracy.
    """
    def __init__(self, species: Species, cv: int = 5) -> None:
        super().__init__(species, cv)
        self.scores = [
            ("F1", f1_score, {"average": "samples"}),
            ("Macro-F1", f1_score, {"average": "macro"}),
            ("Micro-F1", f1_score, {"average": "micro"}),
            ("HL", hamming_loss, {}),
            ("SA", accuracy_score, {})]

        self.score_names = [x[0] for x in self.scores]
        self.methods = ["lp", "knn", "hf", "rf", "gb", "svc", "ecc", "mlknn"]
        self.result = pd.DataFrame(index=self.score_names, columns=self.methods)
        self.stds = pd.DataFrame(index=self.score_names, columns=self.methods)
        self.k, self.proportions = 2, [0.8, 0.2]

    def evaluate(self, nshuffle: int) -> None:
        """
        Evaluate the classification model using stratified sampling method.
        @param nshuffle: the number of times to shuffle the data.
        """
        print(f">>>Proportions for stratified sampling: {self.proportions}")
        result_nsh = {key: pd.DataFrame(columns=self.score_names, index=range(nshuffle))
                      for key in self.result.columns}
        # kf = KFold(n_splits=self.cv, shuffle=True, random_state=29)
        for fold in range(nshuffle):
            # for fold, (train_indices, test_indices) in enumerate(kf.split(self.species.X), 1):
            self.species.X, self.species.Y = shuffle(self.species.X, self.species.Y, random_state=23)
            subsets = iterative_stratification(self.species.Y, k=self.k, proportions=self.proportions)
            train_indices, test_indices = subsets
            for method in self.methods:
                func_name = f"{method}_predict"
                if hasattr(self, func_name):
                    func = getattr(self, func_name)
                    preds = func(train_indices, test_indices)
                    for metric, scoring_func, kwargs in self.scores:
                        result_nsh[method].loc[fold, metric] = round(
                            scoring_func(self.species.Y[test_indices], preds, **kwargs), 3)
                else:
                    raise AttributeError(f"Method {func_name} is not defined in the class.")

        for method_name in list(self.methods):
            self.result[method_name] = result_nsh[method_name].mean()
            self.stds[method_name] = result_nsh[method_name].std()


    def printResults(self):
        print(f">>>Results: \n{self.result.to_string()}")
        print(f"Standard deviations: \n{self.stds.to_string()}")


    def ecc_predict(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Prediction using Classifier Chains with a random forest base model.
        @param train_indices
        @param test_indices
        @return predictions
        """
        base_model = RandomForestClassifier(random_state=49)
        ecc = EnsembleClassifierChains(base_estimator=base_model, n_chains=2, random_state=23)
        Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        ecc.fit(self.species.X[train_indices], Y_train)
        soft_labels = ecc.predict(self.species.X[test_indices])
        preds = lco(soft_labels, Y_test)  # apply one of the thresholding strategy: lco, cmn, or basic at 0.5
        return preds

    def lp_predict(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Prediction using label powerset with a Random Forest base model.
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

    def mlknn_predict(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Prediction using ml-knn.
        @param train_indices
        @param test_indices
        @return predictions
        """
        mlknn = MLkNN(k=2, s=0.5)
        Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        mlknn.fit(self.species.X[train_indices], Y_train)
        preds = mlknn.predict(self.species.X[test_indices])
        return preds

    def hf_predict(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Prediction using harmonic function.
        @param train_indices
        @param test_indices
        @return predictions
        """
        Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        sigma, nn = 0.5, 5
        X_dist_squared = dist_matrix(self.species.X) ** 2
        soft_labels = hf(X_dist_squared, Y_train, train_indices, test_indices, nn=nn, sigma=sigma)
        preds = lco(soft_labels, Y_test)  # apply one of the thresholding strategy: lco, cmn, or basic at 0.5
        return preds

    def cm_predict(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Prediction using consistency method.
        @param train_indices
        @param test_indices
        @return predictions
        """
        Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        sigma, reg = 2, 0.8
        X_dist_squared = dist_matrix(self.species.X) ** 2
        soft_labels = cm(X_dist_squared, np.array(self.species.Y_top), train_indices=train_indices, sigma=sigma,
                         reg=reg)
        preds = basic(soft_labels, Y_test)  # apply one of the thresholding strategy: lco, cmn, or basic at 0.5
        return preds

    def knn_predict(self, train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Predict using k-nearest neighbour.
        We choose very small number of neighbour between 1 and 3.
        @param train_indices
        @param test_indices
        @return predictions
        """
        Y_train, Y_test = self.species.Y[train_indices], self.species.Y[test_indices]
        preds = np.zeros_like(Y_test)
        normalized_X = StandardScaler().fit_transform(self.species.X)
        for s in range(Y_train.shape[1]):
            knn = KNeighborsClassifier(n_neighbors=1).fit(normalized_X[train_indices], Y_train[:, s])
            preds[:, s] = knn.predict(normalized_X[test_indices])
        return preds

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
