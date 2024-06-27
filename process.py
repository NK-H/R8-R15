import pickle
import numpy as np
from collections import defaultdict

TRAIN_DATA_PATH = './data/train.txt'
TEST_DATA_PATH = './data/test.txt'
NODE_INDEX_PATH = './data/node_idx.pkl'
TRAIN_USER_DATA_PKL_PATH = './data/train_user.pkl'
TRAIN_ITEM_DATA_PKL_PATH = './data/train_item.pkl'
TEST_DATA_PKL_PATH = './data/test.pkl'

TRAIN_VALID_SPLIT_RATIO = 0.85


def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle_data(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def create_node_index(train_data_path):
    all_nodes = set()
    with open(train_data_path, 'r') as f:
        while (line := f.readline()) != '':
            _, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = f.readline()
                item_id, _ = map(int, line.strip().split())
                all_nodes.add(item_id)
    node_index = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    return node_index


def process_train_data(train_data_path, node_index):
    user_data, item_data = defaultdict(list), defaultdict(list)
    with open(train_data_path, 'r') as f:
        while (line := f.readline()) != '':
            user_id, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = f.readline()
                item_id, score = line.strip().split()
                item_id, score = int(item_id), float(score)
                score = score / 10
                user_data[user_id].append([node_index[item_id], score])
                item_data[node_index[item_id]].append([user_id, score])
    return user_data, item_data


def process_test_data(test_data_path):
    test_data = defaultdict(list)
    with open(test_data_path, 'r') as f:
        while (line := f.readline()) != '':
            user_id, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = f.readline()
                item_id = int(line.strip())
                test_data[user_id].append(item_id)
    return test_data


def split_train_valid_data(user_data, split_ratio=TRAIN_VALID_SPLIT_RATIO, shuffle=True):
    train_data, valid_data = defaultdict(list), defaultdict(list)
    for user_id, items in user_data.items():
        if shuffle:
            np.random.shuffle(items)
        train_data[user_id] = items[:int(len(items) * split_ratio)]
        valid_data[user_id] = items[int(len(items) * split_ratio):]
    return train_data, valid_data


def process_main():
    print('Creating item index...')
    node_index = create_node_index(TRAIN_DATA_PATH)
    save_pickle_data(NODE_INDEX_PATH, node_index)
    print('Done!')

    print('Processing data...')
    user_data, item_data = process_train_data(TRAIN_DATA_PATH, node_index)
    save_pickle_data(TRAIN_USER_DATA_PKL_PATH, user_data)
    save_pickle_data(TRAIN_ITEM_DATA_PKL_PATH, item_data)
    test_data = process_test_data(TEST_DATA_PATH)
    save_pickle_data(TEST_DATA_PKL_PATH, test_data)
    print('Done!')


