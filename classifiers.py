import typing
import lightgbm as lgb

from metrics import evaluate


def model_builder(model_name='lgbm'):
    if model_name == 'lgbm':
        return lgbm
    else:
        return lgbm


def lgbm(trainX, trainY, testX, testY) -> typing.Tuple[float]:
    """
    train and test with lightGBM
    :param trainX:
    :param trainY:
    :param testX:
    :param testY:
    :return:
    """
    train_data = lgb.Dataset(trainX, label=trainY)
    params = {
        'learning_rate': .1,
        'lambda_l1': .1,
        'max_depth': 4,
        'objective': 'binary',
    }
    clf = lgb.train(params, train_data)
    predY = clf.predict(testX)
    predY = [1 if pred > .5 else 0 for pred in predY]       # binarize
    return evaluate(testY, predY)