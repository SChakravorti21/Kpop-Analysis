import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing, feature_selection
from sklearn.model_selection import cross_val_score, GridSearchCV

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

input_path = os.path.join("data", "playlist_features.json")
features = pd.read_json(input_path)

X = features[FEATURES]
y = features["playlist"].map(CATEGORIES)

pipe = Pipeline([
    ("one-hot", ColumnTransformer([
        ("key_category", preprocessing.OneHotEncoder(handle_unknown="ignore"), ["key"])
    ], remainder="passthrough")),
    ("scale", preprocessing.MinMaxScaler()),
    ("select", feature_selection.SelectKBest()),
    ("model", GradientBoostingClassifier())
])

param_grid = {
    "select__k": [5, 6, 7, 8, 9],
    "model__n_estimators": [50, 60, 70, 80, 100, 150, 175, 200],
    # "model__min_samples_split": [2, 0.05, 0.1, 0.2],
    "model__max_depth": [1, 2]
}

search = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1, verbose=10).fit(X, y)
clf = search.best_estimator_
print("\n\nAccuracy: %0.2f " % (search.best_score_))
print(clf)