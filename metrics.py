"""
evaluation metrics
pd = recall = TP / (TP + FN)
pf = FP / (FP + TN)
auc
p = precision = TP / (TP + FP)
F = F-score = 2 * precision * recall / (precision + recall)
"""

from sklearn.metrics import roc_auc_score


def evaluate(testY, predY):
    """
    evaluate by pd, pf, auc, p, F
    :param testY: true 0/1 labels of test samples
    :param predY: predicted 0/1 labels of test samples
    :return: dict of pd, pf, auc, p, F values, format: {$METRIC_NAME: $METRIC_VALUE}
    """
    tp = fp = tn = fn = 0
    for index in range(len(testY)):
        pred = predY[index]
        gt = testY[index]
        if pred == 1 and gt == 1:
            tp += 1
        if pred == 1 and gt == 0:
            fp += 1
        if pred == 0 and gt == 1:
            tn += 1
        if pred == 0 and gt == 0:
            fn += 1
    if tp + fn != 0:
        pd = tp / (tp + fn)
    else:
        pd = 0
    if fp + tn != 0:
        pf = fp / (fp + tn)
    else:
        pf = 0
    if tp + fp == 0:
        p = 0
    else:
        p = tp / (tp + fp)
    if pd + p == 0:
        f = 0
    else:
        f = 2 * pd * p / (pd + p)
    try:
        auc = roc_auc_score(testY, predY)
    except ValueError:
        auc = 1.0
    return {'pd': pd, 'pf': pf, 'auc': auc, 'p': p, 'F': f}