"""
default running environment: Windows 10
"""

import typing
import numpy as np

from datasets.pcap_loaders import load_train_test_data
from classifiers import model_builder


def experiments(trainX, trainY, testX, testY) -> typing.Dict:
    """
    PU learning experiments with positive & unlabeled data for training and positive & negative data for testing
    :param trainX:
    :param trainY:
    :param testX:
    :param testY:
    :return: {"pd": tp / (tp + fn),
                "pf": fp / (fp + tn),
                "auc": roc_auc_score,
                "p": tp / (tp + fp),
                "F": 2 * pd * p / (pd + p)}
                !mention that in evaluation metrics, `p` in `tp`/`fp` means abnormal packages, which are `N` in `testN`
                and `n` in `tn`/`fn` means normal packages, which are `P` in `trainP`/`testP`
    """
    EXPERIMENT_TIMES = 5
    results = {}
    trainY = np.array([0 if y == 1 else 1 for y in trainY])  # reverse positive and negative
    testY = np.array([0 if y == 1 else 1 for y in testY])  # reverse positive and negative
    for model_name in ('lgbm', ):
        model = model_builder(model_name)
        mean_metrics = {'pd': 0.0, 'pf': 0.0, 'auc': 0.0, 'p': 0.0, 'F': 0.0}
        for idx in range(EXPERIMENT_TIMES):
            tmp = model(trainX, trainY, testX, testY)
            mean_metrics = {metric_name: value + tmp[metric_name] for metric_name, value in mean_metrics.items()}
        results[model_name] = {metric_name: value / EXPERIMENT_TIMES for metric_name, value in mean_metrics.items()}
    return results


if __name__ == '__main__':
    data_root = 'D:\\wyxData\\data\\pcap'
    trainX, trainY, testX, testY = load_train_test_data(data_root, drop_timestamp=True)
    print('training feature shape: {}'.format(trainX.shape))
    print('training label shape: {}'.format(len(trainY)))
    print('testing feature shape: {}'.format(testX.shape))
    print('testing label shape: {}'.format(len(testY)))
    results = experiments(trainX, trainY, testX, testY)
    print(results)

