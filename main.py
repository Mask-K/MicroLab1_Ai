from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import libKNN
import myKNN
import mySVM

iris_data = pd.read_csv('IRIS.csv')
X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


n = int(input("1-SVM method, 0-KNN method => "))

if n == 0:
    n = int(input("1-lib method, 0-my method => "))
    if n == 1:
        libKNN.runner(X_train, X_test, y_train, y_test)
    else:
        myKNN.runner(X_train, X_test, y_train, y_test)
else:
    mySVM.runner(X_train, X_test, y_train)


