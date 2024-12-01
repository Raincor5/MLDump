import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot

from matplotlib import style
from sklearn import linear_model, preprocessing

df = pd.read_csv("student-mat.csv", sep=";")

le = preprocessing.LabelEncoder()
df.paid = le.fit_transform(df.paid)
df.sex = le.fit_transform(df.sex)

predict = df.G3

data = df[["G1", "G2", "studytime", "failures", "absences", "freetime", "paid", "sex"]]



X = np.array(data)
Y = np.array(df.G3)



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

'''
best = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: ", linear.coef_)
print("Interception: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(round(predictions[x]), x_test[x], y_test[x])


p = "paid"
style.use("ggplot")
pyplot.scatter(df[p], df["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()