"""
default running environment: Windows 10
"""

import typing
import numpy as np

from datasets.pcap_loaders import load_train_test_data
from classifiers import model_builder


def experiments(trainX, trainY, testX, testY) -> typing.Dict[str, float]:
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
    results = {}
    for model_name in ("lgbm",):
        model = model_builder(model_name)
        trainY = np.array([0 if y == 1 else 1 for y in trainY])         # reverse positive and negative
        testY = np.array([0 if y == 1 else 1 for y in testY])           # reverse positive and negative
        results[model_name] = model(trainX, trainY, testX, testY)
    return results


if __name__ == '__main__':
    data_root = 'D:\\wyxData\\data\\pcap'
    trainX, trainY, testX, testY = load_train_test_data(data_root, negative_split=0.5, drop_timestamp=True)
    print('training feature shape: {}'.format(trainX.shape))
    print('training label shape: {}'.format(len(trainY)))
    print('testing feature shape: {}'.format(testX.shape))
    print('testing label shape: {}'.format(len(testY)))
    results = experiments(trainX, trainY, testX, testY)
    print(results)

