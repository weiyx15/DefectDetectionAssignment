"""
data loaders for industry network malicious traffic detection dataset
"""

import typing
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


if __name__ == '__main__':
    data_root = 'D:\\wyxData\\data\\pcap'
    all_fn = os.path.join(data_root, 'all', 'IPv6_TCP.csv')
    normal_fn = os.path.join(data_root, 'normal', 'IPv6_TCP.csv')
    input_fns = (all_fn, normal_fn)
    for fn in input_fns:
        data = IPv6TCPLoader.load_features(fn)
        print(data.head(10))
        print(data.dtypes)
