import typing
import numpy as np
import lightgbm as lgb
from sklearn import svm, neural_network, naive_bayes, ensemble, linear_model

from metrics import evaluate


def model_builder(model_name='lgbm'):
    if model_name == 'lgbm':
        return lgbm
    elif model_name == 'lgbm_pu':
        return lgbm_pu
    elif model_name == 'gbdt':
        return gbdt
    elif model_name == 'svm':
        return svc
    elif model_name == 'mlp':
        return mlp
    elif model_name == 'bayes':
        return bayes
    elif model_name == 'rf':
        return random_forest
    elif model_name == 'lr':
        return logsitic
    else:
        return lgbm              # default classifier


def lgbm(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with lightGBM
    Weight for positive samples varies for unbalanced data or unlabeled data
    """
    train_data = lgb.Dataset(trainX, label=trainY)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': 10,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'scale_pos_weight': 1.4,
        'min_data': 1,
        'min_data_in_bin': 1
    }

    clf = lgb.train(params, train_data)
    scoreY = clf.predict(testX)
    predY = [1 if score > .5 else 0 for score in scoreY]       # binarize
    return evaluate(testY, scoreY, predY)


def lgbm_pu(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with lightGBM for PU learning, where training data only contains positive and unlabeled data
    1. Train g(x) to classify positive versus unlabeled data
    2. Calculate c = mean(g(x)), x is positive samples
    3. f(x) = g(x) / c, where f(x) is the classifier for positive versus negative data
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
    scoreY = clf.predict(testX)
    pos = [score for score in scoreY if score > .5]
    coeff = sum(pos) / len(pos)
    threshold = .5 * coeff
    print('[lgbm_pu][debug] threshold = {}'.format(threshold))
    predY = [1 if score > threshold else 0 for score in scoreY]  # binarize
    return evaluate(testY, scoreY, predY)


def gbdt(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with gradient boosting decision tree
    """
    clf = ensemble.GradientBoostingClassifier()
    try:
        clf.fit(trainX, trainY)
        predY = clf.predict(testX)
        scoreY = clf.predict_proba(testX)
        if len(scoreY.shape) == 2 and scoreY.shape[1] == 2:
            scoreY = np.max(scoreY, axis=1)
    except ValueError:
        predY = np.ones((testY.shape[0],))
        scoreY = predY
    if int(np.sum(predY)) == 0:
        predY = np.ones((testY.shape[0],))
        scoreY = predY
    return evaluate(testY, scoreY, predY)


def svc(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with svm classifier
    """
    clf = svm.SVC(gamma='auto')
    try:
        clf.fit(trainX, trainY)
        predY = clf.predict(testX)
        scoreY = predY
    except ValueError:
        predY = np.ones((testY.shape[0],))
        scoreY = predY
    if int(np.sum(predY)) == 0:
        predY = np.ones((testY.shape[0],))
        scoreY = predY
    return evaluate(testY, scoreY, predY)


def mlp(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with neural network classifier
    """
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(32,), learning_rate_init=0.0001, max_iter=1000)
    clf.fit(trainX, trainY)
    scoreY = clf.predict_proba(testX)
    if len(scoreY.shape) == 2 and scoreY.shape[1] == 2:
        scoreY = np.max(scoreY, axis=1)
    predY = clf.predict(testX)
    return evaluate(testY, scoreY, predY)


def bayes(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with naive bayes
    """
    clf = naive_bayes.GaussianNB()
    clf.fit(trainX, trainY)
    scoreY = clf.predict_proba(testX)
    if len(scoreY.shape) == 2 and scoreY.shape[1] == 2:
        scoreY = np.max(scoreY, axis=1)
    predY = clf.predict(testX)
    return evaluate(testY, scoreY, predY)


def random_forest(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with random forest
    """
    clf = ensemble.RandomForestClassifier()
    clf.fit(trainX, trainY)
    scoreY = clf.predict_proba(testX)
    if len(scoreY.shape) == 2 and scoreY.shape[1] == 2:
        scoreY = np.max(scoreY, axis=1)
    predY = clf.predict(testX)
    return evaluate(testY, scoreY, predY)


def logsitic(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with naive bayes
    """
    clf = linear_model.LogisticRegression()
    try:
        clf.fit(trainX, trainY)
        predY = clf.predict(testX)
        scoreY = clf.predict_proba(testX)
        if len(scoreY.shape) == 2 and scoreY.shape[1] == 2:
            scoreY = np.max(scoreY, axis=1)
    except ValueError:
        predY = np.ones((testY.shape[0],))
        scoreY = predY
    if int(np.sum(predY)) == 0:
        predY = np.ones((testY.shape[0],))
        scoreY = predY
    return evaluate(testY, scoreY, predY)