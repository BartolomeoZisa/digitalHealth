import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score
import warnings
from pandas.errors import PerformanceWarning
from sktime.clustering.k_means import TimeSeriesKMeans
# This silences the specific fragmentation warning from Pandas
warnings.filterwarnings('ignore', category=PerformanceWarning)




class AlignedTimeSeriesKMeans(ClassifierMixin, BaseEstimator):
    
    # FIXED: Explicitly tell Scikit-Learn this pipeline step is a CLASSIFIER
    # This stops it from throwing the "Got a regressor with response_method=predict_proba" error
    _estimator_type = "classifier" 

    def __init__(self, n_clusters=2, init_algorithm='kmeans++', metric='euclidean', 
                 distance_params=None, averaging_method='mean', random_state=42):
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
        
        # Align labels
        preds = self.clusterer_.predict(X)
        if accuracy_score(y, preds) < 0.5:
            self.flip_labels_ = True
        else:
            self.flip_labels_ = False
            
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        preds = self.clusterer_.predict(X)
        if self.flip_labels_:
            preds = 1 - preds
        return preds

    def predict_proba(self, X):
        # Generates binary probability outputs [0.0, 1.0] to satisfy ROC-AUC requirements
        preds = self.predict(X)
        probs = np.zeros((len(preds), len(self.classes_)))
        probs[np.arange(len(preds)), preds] = 1.0
        return probs


from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np

# ==========================================
# K-MEDOIDS CLASSIFIER WRAPPER
# ==========================================
class AlignedTimeSeriesKMedoids(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"  # ensures scikit-learn treats it as a classifier

    def __init__(self, n_clusters=2, init_algorithm='random', metric='euclidean',
                 distance_params=None, random_state=42):
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

        # Align cluster labels with true labels
        preds = self.clusterer_.predict(X)
        self.flip_labels_ = accuracy_score(y, preds) < 0.5
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        preds = self.clusterer_.predict(X)
        if self.flip_labels_:
            preds = 1 - preds
        return preds

    def predict_proba(self, X):
        preds = self.predict(X)
        probs = np.zeros((len(preds), len(self.classes_)))
        probs[np.arange(len(preds)), preds] = 1.0
        return probs



from sktime.classification.distance_based import ShapeDTW

# ==========================================
# 4. SHAPE-DTW CLASSIFIER WRAPPER
# ==========================================
# ==========================================
# 4. SHAPE-DTW CLASSIFIER WRAPPER (FIXED)
# ==========================================
class ShapeDTWClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, n_neighbors=1, shape_descriptor_function='raw'): # Removed subsequence_distance
        self.n_neighbors = n_neighbors
        self.shape_descriptor_function = shape_descriptor_function

    def fit(self, X, y):
        # We pass only the arguments accepted by sktime's ShapeDTW
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
        # ShapeDTW might not implement predict_proba natively depending on version
        # If it fails, use the same logic as your KMeans wrapper
        try:
            return self.clf_.predict_proba(X)
        except AttributeError:
            preds = self.predict(X)
            probs = np.zeros((len(preds), len(self.classes_)))
            probs[np.arange(len(preds)), preds.astype(int)] = 1.0
            return probs
