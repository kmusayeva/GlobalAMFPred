"""
Performs binary-relevance classification.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
from .species_classification import *


class BRClassification(SpeciesClassification):
    def __init__(self, species: Species, metric: str, cv: int = 5) -> None:
        super().__init__(species)
        self.metric = metric
        self.cv = cv  # Assuming cross-validation splits
        self.methods = ["knn", "rf", "gb", "lr"]
        cols = ['best_score', 'best_params']
        self.result = {key: pd.DataFrame(columns=cols, index=self.species_names)
                       for key in self.methods}

    def knn_predict(self, k: int = 5):
        self.species.min_max_norm_env_vars()
        params = {'n_neighbors': range(1, k)}
        for s in self.species.Y_top:
            y = self.species.Y_top[s]
            grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, cv=self.cv, scoring=self.metric,
                                       n_jobs=-1)
            grid_search.fit(self.species.X, y)
            self.result["knn"]['best_score'][s] = grid_search.best_score_.round(2)
            self.result["knn"]['best_params'][s] = grid_search.best_params_

    def rf_predict(self, max_trees: int = 20, max_depth: int = 20, min_sample: int = 10):
        params = {
            'n_estimators': range(5, max_trees, 5),
            'max_depth': range(1, max_depth),
            'min_samples_split': range(2, min_sample),
            'min_samples_leaf': [1, 2, 4]
        }
        for s in self.species.Y_top:
            y = self.species.Y_top[s]
            grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=23), param_grid=params, cv=self.cv,
                                       scoring=self.metric, n_jobs=-1)
            grid_search.fit(self.species.X, y)
            self.result["rf"]['best_score'][s] = grid_search.best_score_.round(2)
            self.result["rf"]['best_params'][s] = grid_search.best_params_

    def gb_predict(self, max_trees: int = 10, max_depth: int = 5, min_sample: int = 10):
        params = {
            'learning_rate': [0.01, 0.1, 1, 10],
            'n_estimators': range(5, max_trees, 5),
            'max_depth': range(1, max_depth),
            'min_samples_split': range(2, min_sample),
            'min_samples_leaf': [1, 2, 4]
        }
        for s in self.species.Y_top:
            y = self.species.Y_top[s]
            grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=params, cv=self.cv,
                                       scoring=self.metric, n_jobs=-1)
            grid_search.fit(self.species.X, y)
            self.result["gb"]['best_score'][s] = grid_search.best_score_.round(2)
            self.result["gb"]['best_params'][s] = grid_search.best_params_

    def lr_predict(self, c: List[float] = None):
        params = {'C': [0.01, 0.1, 1, 10, 100]} if c is None else {'C': c}
        for s in self.species.Y_top:
            y = self.species.Y_top[s]
            grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=params, cv=self.cv,
                                       scoring=self.metric, n_jobs=-1)
            grid_search.fit(self.species.X, y)
            self.result["lr"]['best_score'][s] = grid_search.best_score_.round(2)
            self.result["lr"]['best_params'][s] = grid_search.best_params_

    def run(self):
        for method in self.methods:
            func_name = f"{method}_predict"
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                func()
        result_df = pd.concat([self.result[name]['best_score'].rename(name) for name in self.methods], axis=1)
        return result_df

