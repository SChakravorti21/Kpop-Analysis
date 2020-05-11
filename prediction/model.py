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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing, feature_selection
from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
    GridSearchCV,
    StratifiedKFold,
    LeaveOneOut
)

CATEGORIES = {
    "Happy":            0,
    "Chill":            0,
    "Sad/Sentimental":  1,
    "Bops":             2,
    "Madness":          2
}

FEATURES    = ["danceability", "energy", "key",
               "loudness", "mode", "speechiness",
               "acousticness", "instrumentalness",
               "liveness", "valence", "tempo", "explicit"]

CONTINUOUS  = ["danceability", "energy",
               "speechiness", "acousticness",
               "liveness", "valence", "tempo"]

class ModelType(Enum):
    GradientBoosting    = "gbt"
    NeuralNetwork       = "nn"
    KNearestNeighbors   = "knn"
    SupportVector       = "svm"


class PlaylistClassifier():
    def __init__(self, model_type: ModelType):
        input_path = os.path.join("data", "playlist_features.json")
        self.features = pd.read_json(input_path)

        self.model_type = model_type
        self.X = self.features[FEATURES]
        self.y = self.features["playlist"].map(CATEGORIES)
        self.model_path = \
            os.path.join("prediction", "models", self.model_type.value)

    def train(self):
        pipe, param_grid = {
            ModelType.GradientBoosting:     self._get_gbt_pipeline(),
            ModelType.NeuralNetwork:        self._get_nn_pipeline(),
            ModelType.KNearestNeighbors:    self._get_knn_pipeline(),
            ModelType.SupportVector:        self._get_svm_pipeline()
        }[self.model_type]

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

        labels = self._get_label_names()
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
        pd.set_option('display.max_colwidth', None)
        print(X.to_string())

    def pca(self):
        # Only do PCA on continuous variables. Categorical
        # values like `key` are misleading since they make
        # it seem like values are really spread out.
        X = self.X[CONTINUOUS]
        X = PCA(n_components=3).fit_transform(X)

        classes = self._get_label_names()
        labels = [classes[label] for label in self.y]

        plot_df = pd.DataFrame({
            "PC 1": X[:,0],
            "PC 2": X[:,1],
            "PC 3": X[:,2],
            "cluster": labels
        })
        g = sns.PairGrid(plot_df, hue="cluster", palette="coolwarm")
        g = g.map(sns.scatterplot, linewidths=0.75, edgecolor="w", s=40)
        g = g.add_legend()
        g.fig.set_size_inches(10, 10)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("PCA on Playlist Dataset", fontsize=16)
        plt.savefig(os.path.join("prediction", "pca.png"))
        plt.close()

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
        without_key = [feature for feature in FEATURES if feature != "key"]

        pipe = Pipeline([
            ("transform", ColumnTransformer([
                ("scale",
                 preprocessing.MinMaxScaler(),
                 without_key),
                ("key_category",
                 preprocessing.OneHotEncoder(handle_unknown="ignore"),
                 ["key"])
            ], remainder="passthrough")),
            # ("select", feature_selection.SelectKBest()),
            ("model", GradientBoostingClassifier())
        ])

        param_grid = {
            # "select__k": [5, 6, 7, 8, 9, 10, 11],
            "model__n_estimators": [75, 100, 125, 150, 200, 250, 300],
            # "model__min_samples_split": [2, 0.05],
            "model__max_depth": [1, 2, 3, 4]
        }

        return pipe, param_grid

    def _get_nn_pipeline(self):
        without_key = [feature for feature in FEATURES if feature != "key"]

        pipe = Pipeline([
            ("transform", ColumnTransformer([
                ("scale",
                 preprocessing.StandardScaler(),
                 without_key),
                ("key_category",
                 preprocessing.OneHotEncoder(handle_unknown="ignore"),
                 ["key"])
            ], remainder="passthrough")),
            ("select", feature_selection.SelectKBest()),
            ("model", MLPClassifier(solver="adam", max_iter=5000,
                                    early_stopping=True))
        ])

        param_grid = {
            "select__k": [4, 5, 6, 7, 8],
            "model__alpha": 10.0 ** -np.arange(5, 12),
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

    def _get_svm_pipeline(self):
        pipe = Pipeline([
            ("scale", preprocessing.MinMaxScaler()),
            ("select", feature_selection.SelectKBest()),
            ("model", SVC(kernel="poly", max_iter=-1, break_ties=True, probability=True))
        ])

        param_grid = {
            "select__k": [4, 5, 6, 8, "all"],
            "model__degree": [3, 4, 5, 6],
            "model__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto"]
        }

        return pipe, param_grid

if __name__ == "__main__":
    command = sys.argv[1]
    model_type = None if len(sys.argv) < 3 else sys.argv[2]
    model_type = ModelType(model_type)
    classifier = PlaylistClassifier(model_type)

    if command == "train":
        classifier.train()
    elif command == "stats":
        classifier.stats()
    elif command == "wrong":
        classifier.misclassified()
    elif command == "pca":
        classifier.pca()
