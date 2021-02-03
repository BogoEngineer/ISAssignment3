import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rand
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    # pd.set_option('display.max_columns', 15)
    # pd.set_option('display.width', None)
    data = pd.read_csv("house_prices_train.csv")

    print(data.head())
    print(data.tail())

    print(data.info())
    print(data.describe())

    # plotting correlation of y and each individual feature
    plt.ylabel("Price")
    for feature in data:
        if feature == "Price": continue
        fig = plt.gcf()
        fig.canvas.set_window_title(feature + " correlation to house price")
        plt.xlabel(feature)
        plt.scatter(data[feature], data.Price)
        plt.show()

    # bedroom_no feature is irrelevant
    data.drop("Bedroom_no", axis=1, inplace=True)

    # normalizing features
    for feature in data:
        if feature == "Price": continue
        data[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())

    X = data[['Year_built', 'Area', 'Bath_no']]
    y = data['Price']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

    linear_reg_from_scratch(x_train, y_train, x_test, y_test)
    sklearn_linear_reg(x_train, y_train, x_test, y_test)


def linear_reg_from_scratch(x_train, y_train, x_test, y_test):
    W = np.array([rand.random_sample() for _ in x_train])
    bias = rand.random()

    gradient_descent(x_train, y_train, W, bias)

    # test model
    prediction = x_test @ W

    # print(y_test)

    # R^2 score
    u = ((y_test - prediction) ** 2).sum()  # residual sum of squares
    v = ((y_test - prediction.mean()) ** 2).sum()  # total sum of squares
    score = 1 - (u / v)

    print("FROM SCRATCH SCORE: ", score)

    print("FROM SCRATCH COEFS", W)


def gradient_descent(features, target, weights, bias):
    """
    m - number of training examples

    :param features: (m, 3)
    :param target:  (m, 1)
    :param weights: (1, 3)
    :param bias: (1, 1)
    :return:
    """

    learning_rate = 0.5
    last_J = float('inf')

    while True:
        prediction = features @ weights + bias  # (m, 1)
        difference = prediction - target  # (m, 1)
        L = difference ** 2  # loss function for each training example - Square Error
        m = len(target)
        J = L.sum() / (2 * m)  # overall cost function for this iteration of GD

        if J >= last_J: break  # if GD doesnt progress with this step, break
        print(J)  # tracking cost function progress
        last_J = J

        # compute gradients
        dW = np.array([difference @ features[x] for x in features])  # (1, 3)
        dW /= m
        weights -= dW * learning_rate

        db = difference.sum()  # (1, 1)
        db /= m
        bias -= db * learning_rate


def sklearn_linear_reg(x_train, y_train, x_test, y_test):
    linearModel = LinearRegression()

    linearModel.fit(x_train, y_train)

    score = linearModel.score(x_test, y_test)

    print("SKLEARN SCORE: ", score)

    print("SKLEARN COEFS: ", linearModel.coef_)


if __name__ == "__main__":
    main()
