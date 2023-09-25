import numpy as np
from sklearn.metrics import accuracy_score

class KNNClassifier:
    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x, y):
        return np.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))

    def _predict(self, item):
        most_frequent_y = [self.y_train._values[x] for x in
                           np.argpartition([self.euclidean_distance(item, i) for i in self.X_train._values], self.k)[0:3]]
        return max(most_frequent_y, key=lambda x: most_frequent_y.count(x))

    def predict(self, X_test):
        return [self._predict(x) for x in X_test._values]

def runner(X_train, X_test, y_train, y_test):
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Predictions:")
    print(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100}")
