from config import config
import random


def read_data():
    data_path = config.DATA_FILE
    x, y = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            fields = line.split('_!_')
            x.append(fields[3])
            y.append(fields[1])
        return x, y


def shuffle_data(X, Y):
    tmp = list(zip(X, Y))
    random.shuffle(tmp)
    X[:], Y[:] = zip(*tmp)
    return X, Y


if __name__ == '__main__':
    x, y = read_data()


