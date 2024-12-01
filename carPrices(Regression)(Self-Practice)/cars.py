import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model, preprocessing

df = pd.read_csv("imports-85.data")

le = preprocessing.LabelEncoder()
df["fuel-type"] = le.fit_transform(df["fuel-type"])
df["wheel-base"] = le.fit_transform(df["wheel-base"])
df["engine-type"] = le.fit_transform(df["engine-type"])
df["engine-size "] = le.fit_transform(df["engine-size"])
df["compression-ratio"] = le.fit_transform(df["compression-ratio"])
df["horsepower"] = le.fit_transform(df["horsepower"])
df["city-mpg"] = le.fit_transform(df["city-mpg"])
df["highway-mpg"] = le.fit_transform(df["highway-mpg"])
df["price"] = le.fit_transform(df["price"])
predict = "price"

data = df[["fuel-type", "wheel-base", "engine-type", "engine-size", "compression-ratio", "horsepower", "city-mpg",
           "highway-mpg"]]

X = np.array(data)
Y = np.array(df.price)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

best = 0
for _ in range(100000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        with open("cars.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("cars.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: ", linear.coef_)
print("Interception: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
print(best)
p = "horsepower"
style.use("ggplot")
pyplot.scatter(df[p], df["price"])
pyplot.xlabel(p)
pyplot.ylabel("123")
pyplot.show()