import numpy as np
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.classification.distance_based import ShapeDTW

from pandas.errors import PerformanceWarning

warnings.filterwarnings('ignore', category=PerformanceWarning)


# =========================================================
# UTILITY: PROPER CLUSTER -> LABEL ALIGNMENT (HUNGARIAN)
# =========================================================
def compute_cluster_mapping(y_true, y_pred):
    """
    Returns mapping: cluster_id -> class_label
    """
    cm = confusion_matrix(y_true, y_pred)

    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return mapping


def apply_mapping(preds, mapping):
    return np.array([mapping[p] for p in preds])


# =========================================================
# 1. TIME SERIES KMEANS WRAPPER (FIXED)
# =========================================================
class AlignedTimeSeriesKMeans(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, n_clusters=2, init_algorithm='kmeans++',
                 metric='euclidean', distance_params=None,
                 averaging_method='mean', random_state=42):

        self.n_clusters = n_clusters
        self.init_algorithm = init_algorithm
        self.metric = metric
        self.distance_params = distance_params
        self.averaging_method = averaging_method
        self.random_state = random_state

    def fit(self, X, y):

        self.clusterer_ = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            init_algorithm=self.init_algorithm,
            metric=self.metric,
            distance_params=self.distance_params,
            averaging_method=self.averaging_method,
            random_state=self.random_state
        )

        self.clusterer_.fit(X)

        preds = self.clusterer_.predict(X)

        # ✅ PROPER ALIGNMENT (NOT FLIPPING)
        self.label_mapping_ = compute_cluster_mapping(y, preds)
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        preds = self.clusterer_.predict(X)
        return apply_mapping(preds, self.label_mapping_)

    def predict_proba(self, X):
        preds = self.predict(X)

        probs = np.zeros((len(preds), len(self.classes_)))
        for i, p in enumerate(preds):
            probs[i, np.where(self.classes_ == p)[0][0]] = 1.0

        return probs


# =========================================================
# 2. TIME SERIES KMEDOIDS WRAPPER (FIXED)
# =========================================================
class AlignedTimeSeriesKMedoids(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, n_clusters=2, init_algorithm='random',
                 metric='euclidean', distance_params=None,
                 random_state=42):

        self.n_clusters = n_clusters
        self.init_algorithm = init_algorithm
        self.metric = metric
        self.distance_params = distance_params
        self.random_state = random_state

    def fit(self, X, y):

        self.clusterer_ = TimeSeriesKMedoids(
            n_clusters=self.n_clusters,
            init_algorithm=self.init_algorithm,
            metric=self.metric,
            distance_params=self.distance_params,
            random_state=self.random_state
        )

        self.clusterer_.fit(X)

        preds = self.clusterer_.predict(X)

        # ✅ PROPER ALIGNMENT
        self.label_mapping_ = compute_cluster_mapping(y, preds)
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        preds = self.clusterer_.predict(X)
        return apply_mapping(preds, self.label_mapping_)

    def predict_proba(self, X):
        preds = self.predict(X)

        probs = np.zeros((len(preds), len(self.classes_)))
        for i, p in enumerate(preds):
            probs[i, np.where(self.classes_ == p)[0][0]] = 1.0

        return probs


# =========================================================
# 3. SHAPE-DTW WRAPPER (OK, MINOR CLEANUP)
# =========================================================
class ShapeDTWClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, n_neighbors=1,
                 shape_descriptor_function='raw'):

        self.n_neighbors = n_neighbors
        self.shape_descriptor_function = shape_descriptor_function

    def fit(self, X, y):

        self.clf_ = ShapeDTW(
            n_neighbors=self.n_neighbors,
            shape_descriptor_function=self.shape_descriptor_function,
        )

        self.clf_.fit(X, y)
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        return self.clf_.predict(X)

    def predict_proba(self, X):
        try:
            return self.clf_.predict_proba(X)
        except AttributeError:
            preds = self.predict(X)

            probs = np.zeros((len(preds), len(self.classes_)))
            for i, p in enumerate(preds):
                probs[i, np.where(self.classes_ == p)[0][0]] = 1.0

            return probs