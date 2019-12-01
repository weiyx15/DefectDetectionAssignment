"""
evaluation metrics
pd = recall = TP / (TP + FN)
pf = FP / (FP + TN)
auc
p = precision = TP / (TP + FP)
F = F-score = 2 * precision * recall / (precision + recall)
"""

from sklearn.metrics import precision_score, recall_score


def evaluate(testY, predY):
    """
    evaluate by pd, pf, auc, p, F
    :param testY: true 0/1 labels of test samples
    :param predY: predicted 0/1 labels of test samples
    :return: dict of pd, pf, auc, p, F values, format: {$METRIC_NAME: $METRIC_VALUE}
    """
    precision = precision_score(testY, predY)
    recall = recall_score(testY, predY)
    F = 2 * precision * recall / (precision + recall)
    return {'p': precision, 'pd': recall, 'F': F}