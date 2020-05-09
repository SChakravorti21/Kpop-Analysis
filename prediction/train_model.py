import os
import sys
import utils
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing, feature_selection
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV

CATEGORIES = {
    "Happy":            0,
    "Chill":            0,
    "Sad/Sentimental":  1,
    "Bops":             2,
    "Madness":          2
}

FEATURES = ["danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "explicit"]

class ModelType(Enum):
    GradientBoosting    = 0
    NeuralNetwork       = 1
    KNearestNeighbors   = 2


class PlaylistClassifier():
    def __init__(self, model_type: ModelType):
        input_path = os.path.join("data", "playlist_features.json")
        self.features = pd.read_json(input_path)

        self.model_type = model_type
        self.X = self.features[FEATURES]
        self.y = self.features["playlist"].map(CATEGORIES)

        if self.model_type == ModelType.GradientBoosting:
            self.model_path = os.path.join("prediction", "models", "gbt")
        elif self.model_type == ModelType.NeuralNetwork:
            self.model_path = os.path.join("prediction", "models", "nn")
        elif self.model_type == ModelType.KNearestNeighbors:
            self.model_path = os.path.join("prediction", "models", "knn")

    def train(self):
        if self.model_type == ModelType.GradientBoosting:
            pipe, param_grid = self._get_gbt_pipeline()
        elif self.model_type == ModelType.NeuralNetwork:
            pipe, param_grid = self._get_nn_pipeline()
        elif self.model_type == ModelType.KNearestNeighbors:
            pipe, param_grid = self._get_knn_pipeline()

        # Perform exhaustive grid search to find the best model
        search = GridSearchCV(pipe, param_grid, cv=10, 
                              n_jobs=-1, verbose=2,
                              scoring="f1_macro")
        search = search.fit(self.X, self.y)
        clf = search.best_estimator_
        
        # Just to see what were the optimal hyperparameters
        print("\n\nAccuracy: %0.2f " % (search.best_score_))
        print(clf[-2])
        print(clf[-1])

        # Save best model to disk
        clf = clf.fit(self.X, self.y)
        utils.makedirs(self.model_path)
        joblib.dump(clf, self.model_path)

    def stats(self):
        clf = self._load_model()
        y_pred = cross_val_predict(clf, self.X, self.y, cv=10)

        confusion_matrix = metrics.confusion_matrix(self.y, y_pred, normalize='true')
        report = metrics.classification_report(self.y, y_pred)
        print(report)        

        labels = ["Happy/Chill", "Sad/Sentimental", "Bops/Madness"]
        fig = plt.figure(figsize=(13, 6))
        ax = sns.heatmap(confusion_matrix, annot=True, fmt="0.2f", 
                         cmap="YlGnBu", vmin=0.0, vmax=1.0,
                         xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        plt.yticks(rotation=0)
        plt.title("Confusion Matrix")
        basename = os.path.basename(self.model_path)
        plt.savefig(os.path.join("prediction", f"confusion_{basename}.png"))
        plt.close()

    def misclassified(self):
        clf = self._load_model()
        y_pred = cross_val_predict(clf, self.X, self.y, cv=10)
        classes = self._get_label_names()

        X = self.features.copy()
        X["pred"] = [classes[label] for label in y_pred]
        X["actual"] = [classes[label] for label in self.y]
        X = X[X["pred"] != X["actual"]]
        X = X[["name", "pred", "actual"]]
        X = X.sort_values(by=["actual", "pred"])
        
        pd.set_option('display.max_rows', len(X))
        pd.set_option('display.max_colwidth', -1)
        print(X)

    def _load_model(self):
        return joblib.load(self.model_path)

    def _get_label_names(self):
        labels_map = {}
        
        for k, v in CATEGORIES.items():
            labels_map.setdefault(v, []).append(k)

        return ["/".join(classes)
                for label, classes
                in labels_map.items()]

    def _get_gbt_pipeline(self):
        pipe = Pipeline([
            ("one-hot", ColumnTransformer([
                ("key_category", 
                 preprocessing.OneHotEncoder(handle_unknown="ignore"), 
                 ["key"])
            ], remainder="passthrough")),
            ("scale", preprocessing.MinMaxScaler()),
            ("select", feature_selection.SelectKBest()),
            ("model", GradientBoostingClassifier())
        ])

        param_grid = {
            "select__k": [5, 6, 7, 8, 9],
            "model__n_estimators": [75, 100, 150, 175, 200, 250],
            "model__min_samples_split": [2, 0.05, 0.1, 0.2],
            "model__max_depth": [1, 2]
        }

        return pipe, param_grid

    def _get_nn_pipeline(self):
        pipe = Pipeline([
            ("one-hot", ColumnTransformer([
                ("key_category", 
                 preprocessing.OneHotEncoder(handle_unknown="ignore"), 
                 ["key"])
            ], remainder="passthrough")),
            ("scale", preprocessing.StandardScaler()),
            ("select", feature_selection.SelectKBest()),
            ("model", MLPClassifier(solver="adam", max_iter=5000, 
                                    early_stopping=True, n_iter_no_change=5))
        ])

        param_grid = {
            "select__k": [4, 6, 8],
            "model__alpha": 10.0 ** -np.arange(7, 12),
            "model__hidden_layer_sizes": [(500,), (500, 100), (500, 500)],
            "model__activation": ["relu", "tanh"],
            "model__learning_rate_init": 10.0 ** -np.arange(1, 6)
        }

        return pipe, param_grid

    def _get_knn_pipeline(self):
        pipe = Pipeline([
            ("scale", preprocessing.MinMaxScaler()),
            ("select", feature_selection.SelectKBest()),
            ("model", KNeighborsClassifier(weights="uniform"))
        ])

        param_grid = {
            "select__k": [4, 6, 8, 10],
            "model__n_neighbors": [1, 2, 5, 8, 10, 15, 20, 30],
            "model__leaf_size": [2, 5, 10, 30, 50]
        }

        return pipe, param_grid

if __name__ == "__main__":
    command = sys.argv[1]
    model_type  = sys.argv[2]

    if model_type == "gbt":
        model_type = ModelType.GradientBoosting
    elif model_type == "nn":
        model_type = ModelType.NeuralNetwork
    elif model_type == "knn":
        model_type = ModelType.KNearestNeighbors

    classifier = PlaylistClassifier(model_type)
    
    if command == "train":
        classifier.train()
    elif command == "stats":
        classifier.stats()
    elif command == "wrong":
        classifier.misclassified()