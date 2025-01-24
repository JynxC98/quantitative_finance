import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of the feature to split on
        self.threshold = threshold        # Threshold value for the split
        self.left = left                  # Left child node
        self.right = right                # Right child node
        self.value = value                # Value at leaf node

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2):
        self.n_trees = n_trees                    # Number of trees in the forest
        self.max_depth = max_depth                  # Maximum depth of each tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_trees):
            sample_idxs = np.random.choice(n_samples, n_samples, replace=True)  # Bootstrap sampling
            X_bootstrap = X[sample_idxs]
            y_bootstrap = y[sample_idxs]
            tree = self.build_tree(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        feature_idxs = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)  # Random feature selection
        best_feature_idx, best_threshold = self.best_split(X, y, feature_idxs)

        left_idxs, right_idxs = self.split(X[:, best_feature_idx], best_threshold)
        left_node = self.build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_node = self.build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(feature_idx=best_feature_idx, threshold=best_threshold, left=left_node, right=right_node)

    def best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for idx in feature_idxs:
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * len(np.unique(y))
            num_right = Counter(classes)

            for i in range(1, len(y)):  # Iterate over possible splits
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                if thresholds[i] == thresholds[i - 1]:
                    continue

                gain = self.information_gain(num_left, num_right)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_threshold = (thresholds[i] + thresholds[i - 1]) / 2

        return split_idx, split_threshold

    def information_gain(self, left_counts, right_counts):
        total_count = sum(left_counts) + sum(right_counts)
        p_left = sum(left_counts) / total_count if total_count > 0 else 0
        p_right = sum(right_counts) / total_count if total_count > 0 else 0

        gain = (self.gini_index(left_counts) * p_left + 
                self.gini_index(right_counts) * p_right)
        
        return 1 - gain  # The lower the gini index the better

    def gini_index(self, counts):
        total_count = sum(counts)
        if total_count == 0:
            return 0
        return 1 - sum((count / total_count) ** 2 for count in counts)

    def split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def most_common_label(self, y):
        most_common = Counter(y).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        tree_preds = [tree.value for tree in self.trees]
        return Counter(tree_preds).most_common(1)[0][0]