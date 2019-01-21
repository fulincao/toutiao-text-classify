import sys
sys.path.append('../')
from config import config
from gensim.models import word2vec
import deal_data
import utils
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train_word2vec():
    logging.info('begin train word2vec ....')
    x, _ = deal_data.read_data()
    utils.generator_words(x)
    sentence = word2vec.Text8Corpus(config.WORDS_FILE)
    model = word2vec.Word2Vec(sentences=sentence, size=config.WORD_VECTOR_LENGTH, min_count=3, workers=4, iter=10)
    model.save(config.WORD2VECTOR_MODEL)
    logging.info('finish train of word2vec ...')


def train_lgb():
    logging.info('begin train lightgbm ....')
    logging.info('load data ....')
    label_information = eval(open(config.DATA_LABEL_FILE, 'r', encoding='utf-8').read().strip())
    x, y = deal_data.read_data()
    w2v = word2vec.Word2Vec.load(config.WORD2VECTOR_MODEL)
    X, Y = [], []
    for i in range(len(x)):
        X.append(utils.content2vec(x[i], w2v))
        Y.append(label_information[int(y[i])][0])

    X, Y = deal_data.shuffle_data(X, Y)
    X_train, Y_train = X[:int(len(X)*0.8)], Y[:int(len(X)*0.8)]
    X_test, Y_test = X[int(len(X) * 0.8):], Y[int(len(X) * 0.8):]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    train_data = lgb.Dataset(X_train, label=Y_train)
    validation_data = lgb.Dataset(X_test, label=Y_test)

    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'verbose': 1,
        'max_depth': 10,
        'num_leaves': 50,
        'objective': 'multiclass',
        'num_class': len(set(Y)),
    }

    clf = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=300)
    logging.info('finish train of lightgbm ....')
    logging.info('test begin ....')
    logging.info('train data accuracy ..... ')
    y_pred = clf.predict(X_train)
    y_pred = [list(x).index(max(x)) for x in y_pred]
    logging.info(accuracy_score(Y_train, y_pred))

    logging.info('validation accuracy .....')
    y_pred = clf.predict(X_test)
    y_pred = [list(x).index(max(x)) for x in y_pred]
    logging.info(accuracy_score(Y_test, y_pred))
    joblib.dump(clf, config.LGB_MODEL)


if __name__ == '__main__':
    train_word2vec()
    train_lgb()
