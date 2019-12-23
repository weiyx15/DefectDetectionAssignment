"""
default running environment: Windows 10
"""

import typing
import numpy as np
import argparse

from datasets.pcap_loaders import load_train_test_data
from classifiers import model_builder


def experiments(trainX, trainY, testX, testY, model_name='lgbm', experiment_times=5) -> typing.Dict:
    """
    PU learning experiments with positive & unlabeled data for training and positive & negative data for testing
    :param trainX:
    :param trainY:
    :param testX:
    :param testY:
    :param model_name: model name string, "lgbm" or "lgbm_weights"
    :param experiment_times: experiment repetition times
    :return: {"pd": tp / (tp + fn),
                "pf": fp / (fp + tn),
                "auc": roc_auc_score,
                "p": tp / (tp + fp),
                "F": 2 * pd * p / (pd + p)}
                !mention that in evaluation metrics, `p` in `tp`/`fp` means abnormal packages, which are `N` in `testN`
                and `n` in `tn`/`fn` means normal packages, which are `P` in `trainP`/`testP`
    """
    trainY = np.array([0 if y == 1 else 1 for y in trainY])  # reverse positive and negative
    testY = np.array([0 if y == 1 else 1 for y in testY])  # reverse positive and negative
    model = model_builder(model_name)
    mean_metrics = {'pd': 0.0, 'pf': 0.0, 'auc': 0.0, 'p': 0.0, 'F': 0.0}
    for _ in range(experiment_times):
        tmp = model(trainX, trainY, testX, testY)
        mean_metrics = {metric_name: value + tmp[metric_name] for metric_name, value in mean_metrics.items()}
    return {metric_name: value / experiment_times for metric_name, value in mean_metrics.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse dataset and model.')
    parser.add_argument('--model', default='lgbm',
                    help='mothod name, "lgbm" or "lgbm_weights"')
    parser.add_argument('--data_root', help='root path of input csv data. '
                                            'Path separators under Windows is `\\\\`, '
                                            'for example, D:\\\\wyxData\\\\data\\\\pcap')
    args = parser.parse_args()

    modol_name = args.model
    data_root = args.data_root
    # data_root = 'D:\\wyxData\\data\\pcap'

    trainX, trainY, testX, testY = load_train_test_data(data_root, drop_timestamp=True)
    print('training feature shape: {}'.format(trainX.shape))
    print('training label shape: {}'.format(len(trainY)))
    print('testing feature shape: {}'.format(testX.shape))
    print('testing label shape: {}'.format(len(testY)))
    results = experiments(trainX, trainY, testX, testY, model_name=modol_name)
    print(results)

