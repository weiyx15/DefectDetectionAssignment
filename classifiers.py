import typing
import numpy as np
import lightgbm as lgb
from sklearn import svm, neural_network

from metrics import evaluate


def model_builder(model_name='lgbm'):
    if model_name == 'lgbm':
        return lgbm
    elif model_name == 'svm':
        return svc
    elif model_name == 'mlp':
        return mlp
    else:
        return svc              # default classifier


def lgbm(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with lightGBM
    """
    train_data = lgb.Dataset(trainX, label=trainY)
    
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 10,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'is_unbalance': 'true',
        'min_data': 1,
        'min_data_in_bin': 1
    }

    clf = lgb.train(params, train_data)
    predY = clf.predict(testX)
    predY = [1 if pred > .5 else 0 for pred in predY]       # binarize
    return evaluate(testY, predY)


def svc(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with svm classifier
    """
    clf = svm.SVC(gamma='scale')
    try:
        clf.fit(trainX, trainY)   
        predY = clf.predict(testX)
    except ValueError:
        predY = np.ones((testY.shape[0],))
    if int(np.sum(predY)) == 0:
        predY = np.ones((testY.shape[0],))
    return evaluate(testY, predY)


def mlp(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with neural network classifier
    """
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(32,), learning_rate_init=0.0001, max_iter=1000)
    clf.fit(trainX, trainY)
    predY = clf.predict(testX)
    return evaluate(testY, predY)