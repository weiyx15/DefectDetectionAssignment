"""
running environment:
    Windows:
        `SYSTEM = Windows`
    Baidu macbook pro:
        conda activate defect
        `SYSTEM = Mac`  
"""

import typing
import json
import numpy as np
from collections import Counter

from classifiers import model_builder
from processors import smote
from data_loaders import get_dataset, load_train_test_data


SYSTEM = 'Windows'


def experiments(config: typing.Dict) -> typing.Dict:
    """
    experiment wrapper with configuration as parameter
    :param config: configuration dict loaded from json file
    :return: dict evaluation metrics of each experiment,
        format: {$DATA_SOURCE_NAME: {$DATA_SET_NAME: {$SPLIT: {$METHOD_NAME: {$METRIC_NAME: $METRIC_VALUE}}}}}
        example: {'CK': {'ant1': {'10%': {'lr': {'precision': .333, 'recall': .333, 'f-score': .333}}}}}
    """
    models = [model_builder(model_name.strip()) for model_name in config['model'].split(',')]
    metrics_dict = {}
    training_rates = ('10%', '20%', '30%')
    for datasource in config['data_sources']:
        datasets = get_dataset(config['data_root'][SYSTEM], datasource)
        mean_dict = [{} for _ in range(len(training_rates))]
        for dataset in datasets:
            trains, tests = load_train_test_data(config['data_root'][SYSTEM], datasource, dataset)
            for split in range(len(training_rates)):
                trainXY, testXY = trains[split], tests[split]
                trainX, trainY = trainXY[:, :-1], np.array(
                    [1 if y >0 else 0 for y in trainXY[:, -1]])      # 1/-1 label to 1/0 label
                testX, testY = testXY[:, :-1], np.array([1 if y > 0 else 0 for y in testXY[:, -1]])
                print('Original dataset shape %s' % Counter(trainY))
                trainX, trainY = smote(trainX, trainY)
                print('Resampled dataset shape %s' % Counter(trainY))
                for model in models:
                    metrics = model(trainX, trainY, testX, testY)
                    metrics_dict.setdefault(datasource, {}).setdefault(dataset, {}) \
                        .setdefault(training_rates[split], {})[str(model).split(' ')[1]] = metrics
                    for (key, val) in metrics.items():
                        mean_dict[split].setdefault(str(model).split(' ')[1], {})[key] \
                            = mean_dict[split].setdefault(str(model).split(' ')[1], {}).setdefault(key, 0.0) + val
        for split in range(len(training_rates)):
            for model in models:
                for (key, val) in mean_dict[split][str(model).split(' ')[1]].items():
                    mean_dict[split][str(model).split(' ')[1]][key] = val / len(datasets)
        metrics_dict[datasource]['mean'] = {training_rates[idx]: mean_dict[idx] for idx in range(len(training_rates))}
    return metrics_dict


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    results = experiments(config)
    for datasource in config['data_sources']:
        print('{}: {}'.format(datasource, results[datasource]['mean']['10%']))
    with open('results.json', 'w') as f:
        json.dump(results, f)
