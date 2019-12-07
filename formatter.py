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

    results['CK']['mean']['10%']['NB'] = {'pd': .3937, 'pf': .6236, 'auc': .7291, 'p': .5081, 'F': .4925}
    results['CK']['mean']['20%']['NB'] = {'pd': .4017, 'pf': .6239, 'auc': .7731, 'p': .4705, 'F': .3521}
    results['CK']['mean']['30%']['NB'] = {'pd': .4189, 'pf': .6399, 'auc': .7734, 'p': .4763, 'F': .3891}

    results['NASA']['mean']['10%']['NB'] = {'pd': .6924, 'pf': .3292, 'auc': .8990, 'p': .6641, 'F': .6984}
    results['NASA']['mean']['20%']['NB'] = {'pd': .7229, 'pf': .3190, 'auc': .8472, 'p': .6049, 'F': .6295}
    results['NASA']['mean']['30%']['NB'] = {'pd': .7319, 'pf': .3374, 'auc': .8294, 'p': .5976, 'F': .6347}

    results['CK']['mean']['10%']['NGSLP'] = {'pd': .4237, 'pf': .4836, 'auc': .7591, 'p': .5132, 'F': .3924}
    results['CK']['mean']['20%']['NGSLP'] = {'pd': .5324, 'pf': .4604, 'auc': .7833, 'p': .5242, 'F': .4212}
    results['CK']['mean']['30%']['NGSLP'] = {'pd': .5523, 'pf': .4399, 'auc': .8234, 'p': .5943, 'F': .4732}

    results['NASA']['mean']['10%']['NGSLP'] = {'pd': .6364, 'pf': .5033, 'auc': .7980, 'p': .4231, 'F': .4284}
    results['NASA']['mean']['20%']['NGSLP'] = {'pd': .6523, 'pf': .4919, 'auc': .8114, 'p': .4133, 'F': .4912}
    results['NASA']['mean']['30%']['NGSLP'] = {'pd': .6419, 'pf': .4774, 'auc': .8544, 'p': .4345, 'F': .5032}

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
                    for metric_name in ('pd', 'pf', 'auc', 'p', 'F'):
                        f.write(',{}'.format(metric_name))
                    f.write('\n')
                    for (model_name, md_val) in sp_val.items():
                        f.write('{},'.format(model_name))
                        for (metric_name, mt_val) in md_val.items():
                            f.write('%.4f,' % mt_val)
                        f.write('\n')


if __name__ == '__main__':
    result_json = 'results.json'
    # compare_formatter(result_json)
    dataset_formatter(result_json)