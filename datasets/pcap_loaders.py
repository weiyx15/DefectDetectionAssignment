"""
pcap train/test data loader functions for model input
"""

import os
import pandas
import numpy as np

from datasets.data_utils import _drop_timestamp, remove_duplicate, train_test_split


def load_train_test_data(data_root: str, negative_split: float=1.0, drop_timestamp: bool=True):
    """
    loaded from csv files in directory `input_csvs`
    :param data_root: data root path
    :param negative_split: ratio of negative samples used as test samples, remaining as unlabelded training samples
    :param drop_timestamp: whether to drop timestamp feature column in numpy feature array
    :return: trainX, trainY, testX, testY (numpy array)
    """
    trainP, trainU, testP, testN = pandas.read_csv(os.path.join(data_root, 'input_csvs', 'trainP.csv')),\
        pandas.read_csv(os.path.join(data_root, 'input_csvs', 'trainU.csv')), \
        pandas.read_csv(os.path.join(data_root, 'input_csvs', 'testP.csv')), \
        pandas.read_csv(os.path.join(data_root, 'input_csvs', 'testN.csv'))
    if abs(negative_split - 1.0) < 1e-6:
        pass        # no split on test negative set
    else:
        trainU = remove_duplicate(trainU, testN)
        trainN, testN = train_test_split(testN, int(testN.shape[0] * negative_split))
        trainU = pandas.concat((trainU, trainN))

    trainP = trainP.to_numpy()
    trainU = trainU.to_numpy()
    testP = testP.to_numpy()
    testN = testN.to_numpy()
    n_trainP, n_trainU, n_testP, n_testN = trainP.shape[0], trainU.shape[0], testP.shape[0], testN.shape[0]
    trainX = np.concatenate((trainP, trainU), axis=0)
    testX = np.concatenate((testP, testN), axis=0)
    trainY = [1 for _ in range(n_trainP)]
    trainY.extend([0 for _ in range(n_trainU)])
    testY = [1 for _ in range(n_testP)]
    testY.extend([0 for _ in range(n_testN)])
    if drop_timestamp:
        trainX = _drop_timestamp(trainX)
        testX = _drop_timestamp(testX)
    trainY = np.array(trainY)
    testY = np.array(testY)
    return trainX, trainY, testX, testY


if __name__ == '__main__':
    # data_root = 'D:\\wyxData\\data\\pcap'                     # Windows
    data_root = '/Users/weiyuxuan/Documents/data/pcap_input'    # Mac
    trainX, trainY, testX, testY = load_train_test_data(data_root, negative_split=0.5)
    print('training feature shape: {}'.format(trainX.shape))
    print('training label shape: {}'.format(len(trainY)))
    print('testing feature shape: {}'.format(testX.shape))
    print('testing label shape: {}'.format(len(testY)))