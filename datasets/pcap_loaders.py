"""
data loaders for industry network malicious traffic detection dataset
"""

import pandas
import numpy as np
import os
import pandas as pd


class PcapLoader:
    """basic class for data loaders of different protocols"""
    @staticmethod
    def load_features(input_fn: str):
        data = pd.read_csv(input_fn)
        return data


class IPv6TCPLoader(PcapLoader):
    """IPv6-TCP loader"""
    def __init__(self):
        super(PcapLoader, self).__init__()
        self.network_type = 'IPv6'
        self.transport_type = 'TCP'
        self._fields = ('time',
                         'total_length',
                         'Ethernet_type',
                         'IP_version',
                         'IPV6_tc',
                         'IPV6_fl',
                         'IPV6_plen',
                         'IPV6_nh',
                         'IPV6_hlim',
                         'TCP_sport',
                         'TCP_dport',
                         'TCP_chksum',
                         'TCP_seq',
                         'TCP_ack',
                         'TCP_dataofs',
                         'TCP_reserved',
                         'TCP_flags',
                         'TCP_window',
                         'TCP_urgptr',
                         'TCP_options_Timestamp_1',
                         'TCP_options_Timestamp_2',
                         'Ethernet_dst_0',
                         'Ethernet_dst_1',
                         'Ethernet_dst_2',
                         'Ethernet_dst_3',
                         'Ethernet_dst_4',
                         'Ethernet_dst_5',
                         'Ethernet_src_0',
                         'Ethernet_src_1',
                         'Ethernet_src_2',
                         'Ethernet_src_3',
                         'Ethernet_src_4',
                         'Ethernet_src_5',
                         'IPV6_src_0',
                         'IPV6_src_1',
                         'IPV6_src_2',
                         'IPV6_src_3',
                         'IPV6_src_4',
                         'IPV6_src_5',
                         'IPV6_src_6',
                         'IPV6_src_7',
                         'IPV6_dst_0',
                         'IPV6_dst_1',
                         'IPV6_dst_2',
                         'IPV6_dst_3',
                         'IPV6_dst_4',
                         'IPV6_dst_5',
                         'IPV6_dst_6',
                         'IPV6_dst_7')
        self._num_fields = 49


def _drop_timestamp(numpy_feature):
    """
    drop timestamp column (which is the first column in the train/test numpy feature array
    :param numpy_feature:
    :return: numpy feature array without first column
    """
    return numpy_feature[:, 1:]


def load_train_test_data(data_root: str, data_format: str, drop_timestamp: bool=True):
    """
    loaded from `input_csvs`/`input_npys`
    :param data_root: data root path
    :param data_format: `csv` or `npy`
    :param drop_timestamp: whether to drop timestamp feature column in numpy feature array
    :return: trainX, trainY, testX, testY (numpy array)
    :raise: ValueError: Invalid data format! Valid formats: `csv` or `npy`
    """
    if data_format == 'csv':
        trainP, trainU, testP, testN = pandas.read_csv(os.path.join(data_root, 'input_csvs', 'trainP.csv')).to_numpy(),\
            pandas.read_csv(os.path.join(data_root, 'input_csvs', 'trainU.csv')).to_numpy(), \
            pandas.read_csv(os.path.join(data_root, 'input_csvs', 'testP.csv')).to_numpy(), \
            pandas.read_csv(os.path.join(data_root, 'input_csvs', 'testN.csv')).to_numpy()
    elif data_format == 'npy':
        trainP, trainU, testP, testN = np.load(os.path.join(data_root, 'input_npys', 'trainP.npy')), \
            np.load(os.path.join(data_root, 'input_npys', 'trainU.npy')), \
            np.load(os.path.join(data_root, 'input_npys', 'testP.npy')), \
            np.load(os.path.join(data_root, 'input_npys', 'testN.npy'))
    else:
        raise ValueError('Invalid data format! Valid formats: `csv` or `npy`')
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
    data_root = 'D:\\wyxData\\data\\pcap'
    all_fn = os.path.join(data_root, 'all', 'IPv6_TCP.csv')
    normal_fn = os.path.join(data_root, 'normal', 'IPv6_TCP.csv')
    trainX, trainY, testX, testY = load_train_test_data(data_root, 'npy')
    pass