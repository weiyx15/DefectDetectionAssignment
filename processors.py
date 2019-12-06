"""
pre-processors for training data, including re-sampling unbalanced data or PU learning
"""

from imblearn.over_sampling import SMOTE


def smote(X, Y):
    """
    SMOTE re-sampling algorithm
    :param X: training features before re-sampling
    :param Y: training labels before re-sampling
    :return: re-sampled training features, re-sampled training labels
    """
    sm = SMOTE()
    return sm.fit_resample(X, Y)
