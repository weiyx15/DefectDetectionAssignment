"""
pre-processors for training data, including re-sampling unbalanced data or PU learning
"""

import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE


def smote(X, Y):
    """
    SMOTE re-sampling algorithm
    :param X: training features before re-sampling
    :param Y: training labels before re-sampling
    :return: re-sampled training features, re-sampled training labels
    """
    vals = Counter(Y).values()
    minv, maxv = min(vals), max(vals)
    if maxv == len(Y):
        return X, Y
    sm = SMOTE(k_neighbors=min(minv-1, 5))
    return sm.fit_resample(X, Y)
