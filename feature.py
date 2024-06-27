import numpy as np
from process import load_pickle_data, save_pickle_data

train_path = './data/train.txt'
train_user_pkl = './data/train_user.pkl'
train_item_pkl = './data/train_item.pkl'


def get_statistics(file_path):
    user_set = set()
    item_set = set()
    rating_count = 0
    average_rating = 0.0
    with open(file_path, 'r') as f:
        while (line := f.readline()) != '':
            user_id, num = map(int, line.strip().split('|'))
            user_set.add(user_id)
            rating_count += num
            for _ in range(num):
                line = f.readline()
                item_id, score = line.strip().split()
                item_id, score = int(item_id), float(score)
                average_rating += score
                item_set.add(item_id)
    average_rating /= rating_count
    return len(user_set), len(item_set), rating_count, average_rating


def get_bias(train_data, train_data_num, average):
    miu = average / 10
    bias = np.zeros(train_data_num, dtype=np.float64)
    for data_id1 in train_data:
        sum = 0.0
        for data_id2, score in train_data[data_id1]:
            sum += score
        bias[data_id1] = sum / len(train_data[data_id1])
    bias -= miu
    return bias


def feature_main():
    print('Getting statistics...')
    user_num, item_num, rating_num, average_rating = get_statistics(train_path)
    print('Number of users:', user_num)
    print('Number of rated items:', item_num)
    print('Number of ratings:', rating_num)
    print('Average rating:', average_rating)

    print('Loading users data...')
    train_user_data = load_pickle_data(train_user_pkl)
    print('Users Data loaded.')
    print('Getting bias of user...')
    bias_user = get_bias(train_user_data, user_num, average_rating)
    save_pickle_data('./data/bx.pkl', bias_user)
    print('Data saved.')

    print('Loading items data...')
    train_item_data = load_pickle_data(train_item_pkl)
    print('Items Data loaded.')
    print('Getting bias of item...', )
    bias_item = get_bias(train_item_data, item_num, average_rating)
    save_pickle_data('./data/bi.pkl', bias_item)
    print('Data saved.')

    print('Done!')


