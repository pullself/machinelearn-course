import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class Bayes:
    def __init__(self):
        self.data = load_iris().data
        self.target = load_iris().target
        self.train_data = []
        self.train_target = []
        self.test_data = []
        self.test_target = []
        self.iris_setosa = []
        self.iris_versicolor = []
        self.iris_virginica = []
        self.mean = []
        self.cov = []
        self.loss = None

    def test(self):
        kf = KFold(5,True,1)
        accuracy = 0
        for train_index, test_index in kf.split(self.data):
            self.train_data.clear()
            self.train_target.clear()
            self.test_data.clear()
            self.test_target.clear()
            for i in train_index:
                self.train_data.append(self.data[i])
                self.train_target.append(self.target[i])
                if self.target[i] == 0:
                    self.iris_setosa.append(self.data[i])
                elif self.target[i] == 1:
                    self.iris_versicolor.append(self.data[i])
                else:
                    self.iris_virginica.append(self.data[i])
            for i in test_index:
                self.test_data.append(self.data[i])
                self.test_target.append(self.target[i])
            self.train(self.iris_setosa, self.iris_versicolor, self.iris_virginica)
            accept = 0
            wrong = 0
            for i in range(len(self.test_target)):
                if self.judge(self.test_data[i]) == self.test_target[i]:
                    accept += 1
                else:
                    wrong += 1
            accuracy += accept/(accept+wrong)
        accuracy /= 5
        print(accuracy)

    def train(self, c1, c2, c3):
        self.mean.append(np.mean(c1, axis=0))
        self.mean.append(np.mean(c2, axis=0))
        self.mean.append(np.mean(c3, axis=0))
        self.cov.append(1/50*np.dot((c1-np.mean(c1)).T,(c1-np.mean(c1))))
        self.cov.append(1/50*np.dot((c2-np.mean(c1)).T,(c2-np.mean(c1))))
        self.cov.append(1/50*np.dot((c3-np.mean(c1)).T,(c3-np.mean(c1))))
        self.loss = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    def judge(self, pos):
        c1_prior = 1/3
        c2_prior = 1/3
        c3_prior = 1/3
        c1_ob = self.__N(0, pos)
        c2_ob = self.__N(1, pos)
        c3_ob = self.__N(2, pos)
        tot = c1_prior*c1_ob+c2_prior*c2_ob+c3_prior*c3_ob
        posterior = [c1_prior*c1_ob/tot,
                     c2_prior*c2_ob/tot, c3_prior*c3_ob/tot]
        R_c1 = 0
        R_c2 = 0
        R_c3 = 0
        for j in range(3):
            R_c1 += self.loss[0][j]*posterior[j]
            R_c2 += self.loss[1][j]*posterior[j]
            R_c3 += self.loss[2][j]*posterior[j]
        if min(R_c1, R_c2, R_c3) == R_c1:
            return 0
        elif min(R_c1, R_c2, R_c3) == R_c2:
            return 1
        else:
            return 2

    def __N(self, c, pos):
        x = 1/((math.pow(2*math.pi, 2)) *
               (math.pow(np.linalg.det(self.cov[c]), 1/2)))
        y1 = np.subtract(pos, self.mean[c])
        y2 = np.linalg.inv(self.cov[c])
        y3 = y1.reshape(y1.shape[0], 1)
        y = np.dot(np.dot(y1, y2), y3)
        res = x*math.exp(-1/2*y)
        return res


if __name__ == '__main__':
    i = Bayes()
    i.test()
    # loss = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0],[1, 1, 1]])
    # print(np.cov(loss, rowvar=False))
    # u = np.mean(loss)
    # print(1/3*np.dot((loss-u).T,(loss-u)))