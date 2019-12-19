"""
Plot data in `pcap_results/lgbm_parameter_tuning.txt`
"""

import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    metric_names = ('pd', 'pf', 'auc', 'p', 'F')

    lgbm_weights = dict()
    lgbm_weights[1.0] = {'pd': 0.2870493991989319, 'pf': 0.0005, 'auc': 0.6430964619492656, 'p': 0.9953703703703703, 'F': 0.4455958549222798}
    lgbm_weights[1.2] = {'pd': 0.7596795727636849, 'pf': 0.0, 'auc': 0.8979959946595459, 'p': 1.0, 'F': 0.8634294385432474}
    lgbm_weights[1.4] = {'pd': 0.9279038718291055, 'pf': 0.0, 'auc': 0.988923230974633, 'p': 1.0, 'F': 0.9626038781163435}
    lgbm_weights[1.6] = {'pd': 0.8024032042723631, 'pf': 0.0, 'auc': 0.8066562082777036, 'p': 1.0, 'F': 0.8903703703703704}
    lgbm_weights[1.8] = {'pd': 0.8531375166889186, 'pf': 0.0, 'auc': 0.8567570093457946, 'p': 1.0, 'F': 0.920749279538905}
    lgbm_weights[2.0] = {'pd': 0.6902536715620827, 'pf': 0.0055, 'auc': 0.7505410547396528, 'p': 0.9791666666666666, 'F': 0.8097102584181676}

    for metric_name in metric_names:
        plt.figure()
        plt.title(metric_name, fontsize=36)
        plt.plot(lgbm_weights.keys(), [value_dict[metric_name] for value_dict in lgbm_weights.values()], linewidth=3)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.savefig(os.path.join('..', 'pcap_results', 'lgbm_weights_{}'.format(metric_name)))