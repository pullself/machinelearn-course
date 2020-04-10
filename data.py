import numpy as np
import matplotlib.pyplot as plt
from math import pow
from sklearn.datasets import load_iris
from iris_vis import Visualization


class Data:
    def __init__(self, arr):
        self.dataset = arr
        self.cov = None
        self.eigenvalue = None
        self.featurevector = None
        self.w1 = None
        self.w2 = None
        self.w = None
        self.newdataset = None

    def ceshi(self):
        print(self.eigenvalue)
        print(self.featurevector)

    def count(self):
        self.cov = np.cov(self.dataset, rowvar=False)
        self.eigenvalue, self.featurevector = np.linalg.eig(self.cov)
        norm = np.linalg.norm(self.featurevector, axis=0)
        self.featurevector = np.divide(self.featurevector, norm)
        self.w1 = self.featurevector.T
        self.w2 = np.empty(shape=[0,len(self.cov)])
        for i in range(len(self.cov)):
            temp = [[]]
            for j in range(len(self.cov)):
                if i == j:
                    temp[0].append(pow(self.eigenvalue[i], -1/2))
                else:
                    temp[0].append(0)
            self.w2 = np.append(self.w2, temp, axis=0)
        self.w = np.dot(self.w2, self.w1)
        t = np.dot(self.w,self.dataset.T)
        self.newdataset = t.T
        print(np.cov(self.newdataset, rowvar=False))
        return self.newdataset


if __name__ == '__main__':
    i = Data(load_iris().data)
    arr = i.count()
    v = Visualization(arr)
    v.vis_2()
