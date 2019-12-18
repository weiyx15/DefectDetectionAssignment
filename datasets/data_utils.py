"""
util functions for data loaders
process raw csvs file after protocol parsing
output processed csv/npy files of train/test features and labels for model input
"""

import os
import typing
import pandas
import numpy as np
from datasets.protocol_loaders import IPv6TCPLoader


def _drop_timestamp(numpy_feature: np.ndarray) -> np.ndarray:
    """
    drop timestamp column (which is the first column in the train/test numpy feature array
    :param numpy_feature:
    :return: numpy feature array without first column
    """
    return numpy_feature[:, 1:]


def drop_equal_columns(input_data: typing.Dict):
    """
    delete columns with all equal values which are not helpful to classification
    :param input_data: dict of trainP, trainU, testP, testN
    :return: dict of trainP, trainU, testP, testN (without columns with all equal values)
    """
    for phase in input_data.keys():
        input_data[phase]['_label'] = phase
    concated = pandas.concat([df for df in input_data.values()])
    equal_col_lst = []          # list of column names with all equal values
    for col_name in concated:
        if concated[col_name].unique().size == 1:
            equal_col_lst.append(col_name)

    print('Dummy Columns: {}'.format(equal_col_lst))
    concated = concated.drop(columns=equal_col_lst)
    for phase in input_data.keys():
        input_data[phase] = concated[concated['_label'] == phase].drop(columns=['_label'])
    return input_data


def remove_duplicate(dataset0: pandas.DataFrame, dataset1: pandas.DataFrame) -> pandas.DataFrame:
    """
    remove samples in `dataset0` which also appear in `dataset1`
    :param dataset0: pandas DataFrame
    :param dataset1: pandas DataFrame
    :return: pandas DataFrame, dataset0 with duplicates removed
    """
    joint = dataset0.merge(dataset1, how='outer', indicator=True)
    return joint[joint['_merge'] == 'left_only'].drop(labels=['_merge'], axis=1)


def train_test_split(dataset: pandas.DataFrame, test_num: int) -> (pandas.DataFrame, pandas.DataFrame):
    """
    random select number of `test_num` samples in `dataset` as test set, with remaining as train set
    :param dataset: whole dataset
    :param test_num: number of test samples
    :return: train_set, test_set
    """
    testset = dataset.sample(test_num)
    trainset = remove_duplicate(dataset, testset)
    return trainset, testset


if __name__ == '__main__':
    data_root = 'D:\\wyxData\\data\\pcap'
    all_fn = os.path.join(data_root, 'all', 'IPv6_TCP.csv')
    normal_fn = os.path.join(data_root, 'normal', 'IPv6_TCP.csv')
    abnormal_fn = os.path.join(data_root, 'abnormal', 'IPv6_TCP.csv')
    all_set = IPv6TCPLoader.load_features(all_fn)
    normal_set = IPv6TCPLoader.load_features(normal_fn)
    test_neg = abnormal_set = IPv6TCPLoader.load_features(abnormal_fn)      # negative data for test
    train_unlabeled = all_set = remove_duplicate(all_set, normal_set)       # unlabeled data for train
    train_pos, test_pos = train_test_split(normal_set, 2000)    # positive data for train & test, |test_cases|=2000

    print('# train P: {}'.format(train_pos.shape[0]))
    print('# train U: {}'.format(train_unlabeled.shape[0]))
    print('# test P: {}'.format(test_pos.shape[0]))
    print('# test N: {}'.format(test_neg.shape[0]))

    input_data = {'trainP': train_pos, 'trainU': train_unlabeled, 'testP': test_pos, 'testN': test_neg}

    input_data = drop_equal_columns(input_data)

    for idx, data in input_data.items():
        data.to_csv(os.path.join(data_root, 'input_csvs', idx + '.csv'))
        np.save(os.path.join(data_root, 'input_npys', idx + '.npy'), data.to_numpy())