from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.random.randn(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * 1 / self.n_iters * self.w)
                else:
                    self.w -= self.lr * (2 * 1 / self.n_iters * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


class mySVM:
    def __init__(self, classes, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.classes = set(classes._values)
        self.classifiers = [(SVM(learning_rate, n_iters), claster) for claster in self.classes]

    def fit(self, X, y):
        index = 0
        for i in self.classes:
            binary_y = np.where(y._values == i, 1, -1)
            self.classifiers[index][0].fit(X._values, binary_y)
            index+=1

    def predict(self, X):
        X = X._values
        scores = np.zeros((X.shape[0], len(self.classes)))
        for i in range(len(self.classes)):
            scores[:, i] = self.classifiers[i][0].predict(X)
        return [self.classifiers[i][1] for i in np.argmax(scores, axis=1)]

def runner(X_train, X_test, y_train):
    svm = mySVM(y_train)
    svm.fit(X_train, y_train)
    print(svm.predict(X_test))


