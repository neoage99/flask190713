import mglearn  # 데이터셋 불러들기
import matplotlib.pyplot as plt  # 분류기는 항상 그래프로 보기
from mglearn.datasets import make_wave
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

"""
 가장 이웃한 3개의 최근접 데이터 포인트를 구한 후, 
 각 포인트 사이의 거리의 역수를 가중치로 타겟값을 산정함.
"""

class KnnModel:
    def __init__(self):
        self.x, self.y = make_wave(n_samples=40)
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, random_state=0, test_size = 0.3)
        self.knn_reg = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        # n_jobs : n_jobs가 사용할 코어의 수. -1은 all(모두 사용)
        self.knn_reg.fit(self.x_train, self.y_train)

    def create_model(self):
        _, axes = plt.subplots(1,3)
        line = np.linspace(-5,5, num=1000)
        line = line.reshape(-1,1)

        # for i, ax in zip([1,3,9], axes.ravel()):


