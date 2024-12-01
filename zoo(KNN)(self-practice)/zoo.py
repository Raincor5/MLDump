import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

df = pd.read_csv("zoo.data")
predict = df.type

X = np.array(df.drop(columns = ["type", "name"]))
Y = np.array(predict)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)

for x in range(len(x_test)):
    print("Predicted: ", predicted[x], "Data: ", x_test[x], "Actual: ", y_test[x])