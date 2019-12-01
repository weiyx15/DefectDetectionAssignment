"""
data loaders for CK & NASA software fault detection dataset
"""

import scipy.io as scio
import typing
import os
import json


def load_train_test_data(data_root: str, datasource: str, dataset: str) -> (typing.Dict, typing.Dict):
    """
    load train and test data
    :param data_root: root path of datasources
    :param datasource: name of data source, 'CK' or 'NASA'
    :param dataset: name of dataset
    :return: (train_data, test_data)
                train_data.shape = (n_trains, n_features + 1)
                test_data.shape = (n_tests, n_features + 1)
                last column indicates labels with -1.0/1.0
    """
    train_data_path = os.path.join(data_root, datasource, datasource + 'Train', dataset + 'train')
    test_data_path = os.path.join(data_root, datasource, datasource + 'Test', dataset + 'test')
    return scio.loadmat(train_data_path)[dataset + 'train'][0], scio.loadmat(test_data_path)[dataset + 'test'][0]


def _get_dataset(data_root: str, datasource: str) -> typing.List[str]:
    """
    go through data source directory and fetch names of datasets
    :param data_root: root path of datasources
    :param datasource: name of data source, 'CK' or 'NASA'
    :return:
    """
    dirs = os.listdir(os.path.join(data_root, datasource, datasource + 'Train'))
    return [data_train[:-9] for data_train in dirs]


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    for datasource in config['data_sources']:
        datasets = _get_dataset(config['data_root'], datasource)
        for dataset in datasets:
            train_data, test_data = load_train_test_data(config['data_root'], datasource, dataset)
            pass