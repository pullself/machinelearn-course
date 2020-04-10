import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import load_iris


class Visualization:
    def __init__(self, arr):
        self.iris = load_iris()
        self.target = self.iris.target
        self.data = arr
        self.iris_setosa = np.hsplit(self.data[:50], 4)
        self.iris_versicolor = np.hsplit(self.data[50:100], 4)
        self.iris_virginica = np.hsplit(self.data[100:150], 4)

    def __pca(self, arr: np.ndarray, n):
        pca = decomposition.PCA(n_components=n)
        res = pca.fit_transform(arr)
        ans = [np.hsplit(res[:50], 2)]
        ans.append(np.hsplit(res[50:100], 2))
        ans.append(np.hsplit(res[100:150], 2))
        print(pca.fit(arr).explained_variance_ratio_)
        return ans

    def vis_2(self):
        ticks = []
        ax = []
        ticks.append(np.arange(4, 8, 0.5))
        ticks.append(np.arange(2, 5, 0.5))
        ticks.append(np.arange(1, 7, 1))
        ticks.append(np.arange(0, 2.5, 0.5))
        fig = plt.figure(figsize=(12, 12))
        for i in range(0, 4):
            for j in range(0, 4):
                ax.append(fig.add_subplot(4, 4, i*4+j+1))
                if i == j:
                    ax[i*4+j].set_xticks([])
                    ax[i*4+j].set_yticks([])
                else:
                    ax[i*4+j].scatter(self.iris_setosa[j],
                                      self.iris_setosa[i], c='b', s=5)
                    ax[i*4+j].scatter(
                        self.iris_versicolor[j], self.iris_versicolor[i], c='g', s=5)
                    ax[i*4+j].scatter(
                        self.iris_virginica[j], self.iris_virginica[i], c='r', s=5)
        plt.show()

    def vis_pca(self):
        data = self.__pca(self.data, 2)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(data[0][0], data[0][1], s=10)
        ax.scatter(data[1][0], data[1][1], s=10)
        ax.scatter(data[2][0], data[2][1], s=10)
        plt.show()


if __name__ == "__main__":
    v = Visualization(load_iris().data)
    v.vis_pca()
