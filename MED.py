import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class MED:
    def __init__(self, model=1):
        self.model = model
        self.iris = load_iris()
        self.iris_linear = self.iris.data[:100]
        self.iris_linear_target = self.iris.target[:100]
        self.iris_nonlinear = self.iris.data[50:150]
        self.iris_nonlinear_target = self.iris.target[50:150]
        self.iris_setosa = np.hsplit(self.iris.data[:50], 4)
        self.iris_versicolor = np.hsplit(self.iris.data[50:100], 4)
        self.iris_virginica = np.hsplit(self.iris.data[100:150], 4)
        if self.model == 1:
            self.train_data, self.test_data, self.train_target, self.test_traget = train_test_split(
                self.iris_linear, self.iris_linear_target, test_size=0.3, train_size=0.7, random_state=1, stratify=self.iris_linear_target)
        else:
            self.train_data, self.test_data, self.train_target, self.test_traget = train_test_split(
                self.iris_nonlinear, self.iris_nonlinear_target, test_size=0.3, train_size=0.7, random_state=1, stratify=self.iris_linear_target)
        self.type = None

    def __hold_out(self, ori):
        train = ori
        test = np.empty(shape=[0, 4])
        num = np.arange(50)
        for j in range(15):
            n = np.random.choice(num, 1)
            index = np.where(num == n)
            test = np.append(test, train[index], axis=0)
            train = np.delete(train, index, axis=0)
            num = np.delete(num, index)
        return train, test

    def __data_handling(self):
        train = np.empty(shape=[0, 4])
        test = np.empty(shape=[0, 4])
        x, y = self.__hold_out(self.iris_linear[:50])
        train = np.append(train, x, axis=0)
        test = np.append(test, y, axis=0)
        x, y = self.__hold_out(self.iris_linear[50:100])
        train = np.append(train, x, axis=0)
        test = np.append(test, y, axis=0)
        return train, test

    def train(self):
        dic = {}
        n = len(self.train_data)/2
        sum1 = np.array([0.0, 0.0, 0.0, 0.0])
        sum2 = np.array([0.0, 0.0, 0.0, 0.0])
        for i in range(70):
            if self.train_target[i] == 0:
                sum1 = np.add(sum1, self.train_data[i])
            else:
                sum2 = np.add(sum2, self.train_data[i])
        dic['0'] = np.divide(sum1, [n])
        dic['1'] = np.divide(sum2, [n])
        self.type = dic

    def judge(self, pos):
        x = np.subtract(pos, self.type['0'])
        dis1 = np.dot(x, x.reshape(x.shape[0], 1))
        x = np.subtract(pos, self.type['1'])
        dis2 = np.dot(x, x.reshape(x.shape[0], 1))
        if (dis1 < dis2)[0]:
            return 0
        else:
            return 1

    def test(self):
        index = {}
        if self.type is None:
            print("未训练")
            return index
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        if self.model == 1:
            for i in range(30):
                res = self.judge(self.test_data[i])
                if res == 0 and self.test_traget[i] == 0:
                    TP += 1
                elif res == 1 and self.test_traget[i] == 0:
                    FN += 1
                elif res == 0 and self.test_traget[i] == 1:
                    FP += 1
                else:
                    TN += 1
        else:
            for i in range(30):
                res = self.judge(self.test_data[i])
                if res == 1 and self.test_traget[i] == 1:
                    TP += 1
                elif res == 2 and self.test_traget[i] == 1:
                    FN += 1
                elif res == 1 and self.test_traget[i] == 2:
                    FP += 1
                else:
                    TN += 1
        index['accuracy'] = (TP+TN)/(TP+TN+FP+FN)
        index['precision'] = TP/(TP+FP)
        index['recall'] = TP/(TP+FN)
        index['specificity'] = TN/(TN+FP)
        index['F1'] = (2*index['precision']*index['recall'])/index['precision']+index['recall']
        return index

    def vis(self):
        ticks = []
        ax = []
        ticks.append(np.arange(4, 8, 0.5))
        ticks.append(np.arange(2, 5, 0.5))
        ticks.append(np.arange(1, 8, 1))
        ticks.append(np.arange(0, 3, 0.5))
        fig = plt.figure(figsize=(12, 12))
        for i in range(0, 4):
            for j in range(0, 4):
                ax.append(fig.add_subplot(4, 4, i*4+j+1))
                if i == j:
                    ax[i*4+j].set_xticks([])
                    ax[i*4+j].set_yticks([])
                else:
                    h = -(self.type['0'][j]-self.type['1'][j]) / \
                        (self.type['0'][i]-self.type['1'][i])
                    x1 = (self.type['0'][j]+self.type['1'][j])/2
                    y1 = (self.type['0'][i]+self.type['1'][i])/2
                    b = y1-x1*h
                    x = np.linspace(ticks[j][0], ticks[j][-1], 100)
                    y = h*x+b
                    ax[i*4+j].plot(x, y, 'r-', lw=3)
                    if self.model == 1:
                        ax[i*4+j].scatter(self.iris_setosa[j],
                                        self.iris_setosa[i], c='b', s=5)
                        ax[i*4+j].scatter(
                            self.iris_versicolor[j], self.iris_versicolor[i], c='g', s=5)
                    else:
                        ax[i*4+j].scatter(
                            self.iris_versicolor[j], self.iris_versicolor[i], c='g', s=5)
                        ax[i*4+j].scatter(
                            self.iris_virginica[j], self.iris_virginica[i], c='r', s=5)
                    ax[i*4+j].set_xticks(ticks[j])
                    ax[i*4+j].set_yticks(ticks[i])
        plt.show()


if __name__ == "__main__":
    i = MED(model=2)
    i.train()
    print(i.test())
    i.vis()
