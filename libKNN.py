from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def runner(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print("Predictions:")
    print(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100}")
