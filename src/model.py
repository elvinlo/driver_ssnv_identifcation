import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

def weighted_gini(y, K):
    n = len(y)
    if n == 0:
        return 0.0
    counts = np.bincount(y, minlength=K+1)[1:]
    p = counts / counts.sum()
    Iw = sum(p[i] * p[j] * abs((i+1)-(j+1))
             for i in range(K) for j in range(K))
    return Iw

class WeightedDecisionTreeClassifier:
    def __init__(self, max_depth=10000, min_samples_split=2, 
                 max_features=None, K=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.K = K
        self.tree = None

    class Node:
        def __init__(self, feature=None, thresh=None,
                     left=None, right=None, value=None):
            self.feature = feature
            self.thresh = thresh
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.max_features = self.max_features or X.shape[1]
        self.tree = self._grow_tree(X, y, 0)
        return self

    def _grow_tree(self, X, y, depth):
        # stopping
        if (depth >= self.max_depth 
            or len(y) < self.min_samples_split 
            or len(np.unique(y)) == 1):
            return self.Node(value=self._leaf_value(y))
        
        feat_idxs = np.random.choice(
            X.shape[1], self.max_features, replace=False)
        parent_impurity = weighted_gini(y, self.K)
        
        best_gain, best_feat, best_thresh = -1, None, None
        for feat in feat_idxs:
            for t in np.unique(X[:, feat]):
                left_mask = X[:, feat] <= t
                if not left_mask.any() or left_mask.all():
                    continue
                ig_left = weighted_gini(y[left_mask], self.K)
                ig_right = weighted_gini(y[~left_mask], self.K)
                n, n_l, n_r = len(y), left_mask.sum(), left_mask.size - left_mask.sum()
                gain = parent_impurity - (n_l/n)*ig_left - (n_r/n)*ig_right
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t
        
        if best_gain <= 0:
            return self.Node(value=self._leaf_value(y))
        
        left_mask = X[:, best_feat] <= best_thresh
        left = self._grow_tree(X[left_mask], y[left_mask], depth+1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth+1)
        return self.Node(feature=best_feat, thresh=best_thresh, left=left, right=right)

    def _leaf_value(self, y):
        counts = Counter(y)
        total = sum(counts.values())
        p = {k: counts[k]/total for k in counts}
        best_k, best_cost = None, float('inf')
        for k in range(1, self.K+1):
            cost = sum(p.get(j,0) * abs(k-j) for j in range(1, self.K+1))
            if cost < best_cost:
                best_cost, best_k = cost, k
        return best_k

    def predict_one(self, x):
        node = self.tree
        while node.value is None:
            if x[node.feature] <= node.thresh:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

class WeightedRandomForestClassifier:
    def __init__(self, n_estimators=50, **tree_kwargs):
        self.n_estimators = n_estimators
        self.tree_kwargs = tree_kwargs
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            tree = WeightedDecisionTreeClassifier(**self.tree_kwargs)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        return self

    def predict(self, X):
        all_preds = np.stack([t.predict(X) for t in self.trees], axis=1)
        return np.round(np.median(all_preds, axis=1)).astype(int)

def train_baseline_binary_rf(X_train, y_train,
                             threshold=7,
                             n_estimators=50,
                             max_depth=None,
                             min_samples_split=2,
                             max_features=None,
                             random_state=1):
    """
    Train a binary RF that treats y>=threshold as positive (1)
    and y==1 as negative (0), dropping all other classes.
    """
    mask = (y_train >= threshold) | (y_train == 1)
    # print(f"\n\nEffective data size {mask.sum()} after discarding samples, originally {len(y_train)} samples.")
    Xb, yb = X_train[mask], y_train[mask]
    yb_bin = (yb >= threshold).astype(int)
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 max_features=max_features,
                                 random_state=random_state)
    clf.fit(Xb, yb_bin)
    return clf