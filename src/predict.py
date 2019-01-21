import sys
sys.path.append('../')
import utils
import logging
from config import config
from sklearn.externals import joblib
from gensim.models import word2vec
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def predict(content):
    vec = utils.content2vec(content, w2v)
    res = lgbm.predict([vec])[0]
    label = list(res).index(max(res))
    for key in label_information.keys():
        if label_information[key][0] == label:
            label = label_information[key][1]
            break
    logging.info(content + '\t' + label)
    return label


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logging.error('input content to predict')
        exit(0)

    w2v = word2vec.Word2Vec.load(config.WORD2VECTOR_MODEL)
    lgbm = joblib.load(config.LGB_MODEL)
    label_information = eval(open(config.DATA_LABEL_FILE, 'r', encoding='utf-8').read().strip())
    # predict(content="王者荣耀好菜怎么办")
    # predict(content="联合国维和遇袭")
    # predict(content="国乒包揽五冠")
    # predict(content="创新不够、售量下滑 苹果股价一路下滑 为何巴菲特仍看好?")
    for content in sys.argv[1:]:
        predict(content)
