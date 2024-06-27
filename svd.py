import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from process import split_train_valid_data, load_pickle_data, save_pickle_data

test_pkl = './data/test.pkl'
bx_pkl = './data/bx.pkl'
bi_pkl = './data/bi.pkl'
idx_pkl = './data/node_idx.pkl'


class SVD:
    def __init__(self, model_path='./model',
                 data_path='./data/train_user.pkl', lr=5e-3,
                 lamda1=1e-2, lamda2=1e-2, lamda3=1e-2, lamda4=1e-2,
                 factor=50):
        self.bx = load_pickle_data(bx_pkl)
        self.bi = load_pickle_data(bi_pkl)
        self.lr = lr
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.lamda3 = lamda3
        self.lamda4 = lamda4
        self.factor = factor
        self.idx = load_pickle_data(idx_pkl)
        self.train_user_data = load_pickle_data(data_path)
        self.train_data, self.valid_data = split_train_valid_data(self.train_user_data)
        self.test_data = load_pickle_data(test_pkl)
        self.globalmean = self.get_globalmean()
        self.Q = np.random.normal(0, 0.1, (self.factor, len(self.bi)))
        self.P = np.random.normal(0, 0.1, (self.factor, len(self.bx)))
        self.model_path = model_path


    def get_globalmean(self):
        score_sum, count = 0.0, 0
        for user_id, items in self.train_user_data.items():
            for item_id, score in items:
                score_sum += score
                count += 1
        return score_sum / count


    def predict(self, user_id, item_id):
        pre_score = self.globalmean + \
            self.bx[user_id] + \
            self.bi[item_id] + \
            np.dot(self.P[:, user_id], self.Q[:, item_id])
        return pre_score


    def loss(self, is_valid=False):
        loss, count = 0.0, 0
        data = self.valid_data if is_valid else self.train_data
        for user_id, items in data.items():
            for item_id, score in items:
                loss += (score - self.predict(user_id, item_id)) ** 2
                count += 1
        # 如果是训练集, 计算正则化项
        if not is_valid:
            loss += self.lamda1 * np.sum(self.P ** 2)
            loss += self.lamda2 * np.sum(self.Q ** 2)
            loss += self.lamda3 * np.sum(self.bx ** 2)
            loss += self.lamda4 * np.sum(self.bi ** 2)
        loss /= count
        return loss


    def rmse(self):
        rmse, count = 0.0, 0
        for user_id, items in self.train_user_data.items():
            for item_id, score in items:
                rmse += (score - self.predict(user_id, item_id)) ** 2
                count += 1
        rmse /= count
        rmse = np.sqrt(rmse)
        return rmse

    def train(self, epochs=10, save=False, load=False):
        if load:
            self.load_weight()
        print('Start training...')
        for epoch in range(epochs):
            for user_id, items in tqdm(self.train_data.items(), desc=f'Epoch {epoch + 1}'):
                for item_id, score in items:
                    error = score - self.predict(user_id, item_id)
                    self.bx[user_id] += self.lr * (error - self.lamda3 * self.bx[user_id])
                    self.bi[item_id] += self.lr * (error - self.lamda4 * self.bi[item_id])
                    self.P[:, user_id] += self.lr * (error * self.Q[:, item_id] - self.lamda1 * self.P[:, user_id])
                    self.Q[:, item_id] += self.lr * (error * self.P[:, user_id] - self.lamda2 * self.Q[:, item_id])
            print(f'Epoch {epoch + 1} train loss: {self.loss():.6f} valid loss: {self.loss(is_valid=True):.6f}')
        print('Training finished.')

        if save:
            self.save_weight()


    def test(self, write_path='./result/result.txt', load=True):
        if load:
            self.load_weight()
        print('Start testing...')
        predict_score = defaultdict(list)
        for user_id, item_list in self.test_data.items():
            for item_id in item_list:
                if item_id not in self.idx:
                    pre_score = self.globalmean * 10
                else:
                    new_id = self.idx[item_id]
                    pre_score = self.predict(user_id, new_id) * 10
                    if pre_score > 100.0:
                        pre_score = 100.0
                    elif pre_score < 0.0:
                        pre_score = 0.0

                predict_score[user_id].append((item_id, pre_score))
        print('Testing finished.')

        def write_result(predict_score, write_path):
            print('Start writing...')
            with open(write_path, 'w') as f:
                for user_id, items in predict_score.items():
                    f.write(f'{user_id}|6\n')
                    for item_id, score in items:
                        f.write(f'{item_id} {score}\n')
            print('Writing finished.')

        if write_path:
            write_result(predict_score, write_path)
        return predict_score


    def load_weight(self):
        print('Loading weight...')
        self.bx = load_pickle_data(self.model_path + '/bx.pkl')
        self.bi = load_pickle_data(self.model_path + '/bi.pkl')
        self.P = load_pickle_data(self.model_path + '/P.pkl')
        self.Q = load_pickle_data(self.model_path + '/Q.pkl')
        print('Loading weight finished.')


    def save_weight(self):
        print('saving weight...')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        save_pickle_data(self.model_path + '/bx.pkl', self.bx)
        save_pickle_data(self.model_path + '/bi.pkl', self.bi)
        save_pickle_data(self.model_path + '/P.pkl', self.P)
        save_pickle_data(self.model_path + '/Q.pkl', self.Q)
        print('done.')



def svd_main():
    svd = SVD()
    svd.train(epochs=10, save=True, load=False)
    svd.test()


