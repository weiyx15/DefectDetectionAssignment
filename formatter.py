"""
format `results.json` to csv files
"""

import json
import typing


def compare_formatter(input_json: str):
    """
    format `results.json` to `${DATASOURCE}_${SPLIT}.csv`
    to compare mean performance of all datasets from `${DATASOURCE}` between methods
    output csv
        file names: ${DATASOURCE}_${SPLIT}.csv
        line: methods
        column: metrics
    :param input_json:
    :param config:
    :return:
    """
    with open(input_json, 'r') as f:
        results = json.load(f)
    for (datasource, ds_val) in results.items():
        for (split, sp_val) in ds_val['mean'].items():
            with open('csvs/{}_{}.csv'.format(datasource, split), 'w') as f:
                for (model_name, md_val) in sp_val.items():
                    for (metric_name, mt_val) in md_val.items():
                        f.write('%.4f,'%mt_val)
                    f.write('\n')


def dataset_formatter(input_json: str):
    """
    format `results.json` to `${DATASET}_{SPLIT}.csv` for dataset detail
    output csv
        file names: ${DATASET}_${SPLIT}.csv
        line: methods
        column: metrics
    :param input_json:
    :return:
    """
    with open(input_json, 'r') as f:
        results = json.load(f)
    for dsr_val in results.values():
        for (dataset, dst_val) in dsr_val.items():
            for (split, sp_val) in dst_val.items():
                with open('csvs/{}_{}.csv'.format(dataset, split), 'w') as f:
                    for (model_name, md_val) in sp_val.items():
                        for (metric_name, mt_val) in md_val.items():
                            f.write('%.4f,' % mt_val)
                        f.write('\n')


if __name__ == '__main__':
    result_json = 'results.json'
    compare_formatter(result_json)
    dataset_formatter(result_json)