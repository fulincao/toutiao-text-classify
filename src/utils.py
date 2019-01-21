import numpy as np
from config import config
import jieba


def content2vec(content, model):
    content = jieba.lcut(content)
    res = np.zeros([config.WORD_VECTOR_LENGTH])
    cnt = 0.00000001
    for word in content:
        if word.strip() not in model:
            continue
        res += model[word.strip()]
        cnt += 1
    res = res / cnt
    return res.astype('float32')


def generator_words(x):
    stopwords_file = config.STOPWORDS_FILE
    stopwords = set()
    words = []
    for line in open(stopwords_file, 'r', encoding='utf-8'):
        stopwords.add(line.strip())
    for content in x:
        content = content.strip()
        words.extend([word for word in jieba.lcut(content) if word not in stopwords])
    open(config.WORDS_FILE, 'w', encoding='utf-8').write(' '.join(words))


if __name__ == '__main__':
    pass

