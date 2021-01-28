import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rand
from sklearn.linear_model import LinearRegression


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

    linear_reg_from_scratch(data)
    sklearn_linear_reg(data)


def linear_reg_from_scratch(data):
    W = np.array([rand.random_sample() for x in data if x != "Price"])
    bias = rand.random()

    split_row = int(data.shape[0] * 0.7)
    train, test = data.iloc[:split_row, :], data.iloc[split_row:, :]
    # print(train.info())
    # print(test.info())

    gradient_descent(train.drop('Price', 1), train.Price, W, bias)

    # test model
    prediction = test.drop('Price', 1) @ W
    test.insert(4, "Prediction", prediction, True)

    print(test)

    # R^2 score
    u = ((test['Price'] - test['Prediction']) ** 2).sum()  # residual sum of squares
    v = ((test['Price'] - test['Price'].mean()) ** 2).sum()  # total sum of squares
    score = 1 - (u / v)

    print("FROM SCRATCH SCORE: ", score)

    print("FROM SCRATCH COEFS", W)


def gradient_descent(features, target, weights, bias):
    learning_rate = 0.15
    last_J = float('inf')

    while True:
        prediction = features @ weights # + bias
        difference = prediction - target
        L = difference ** 2  # loss function for each training example - Square Error
        m = len(target)
        J = L.sum() / (2 * m)  # overall cost function for this iteration of GD

        print(J) # tracking cost function progress

        # compute gradients
        dW = np.array([difference @ features[x] for x in features])
        dW /= m
        # print("DW: ", dW)
        weights -= dW * learning_rate

        # db = difference.sum()
        # db /= m
        # bias -= db * learning_rate

        if J >= last_J: break  # if GD doesnt progress with this step, break
        last_J = J


def sklearn_linear_reg(data):
    linearModel = LinearRegression()

    split_row = int(data.shape[0] * 0.7)
    train, test = data.iloc[:split_row, :], data.iloc[split_row:, :]

    X = train[['Year_built', 'Area', 'Bath_no']]
    Y = train['Price']
    linearModel.fit(X, Y)

    X = test[['Year_built', 'Area', 'Bath_no']]
    Y = test['Price']
    score = linearModel.score(X, Y)

    print("SKLEARN SCORE: ", score)

    print("SKLEARN COEF: ", linearModel.coef_)


if __name__ == "__main__":
    main()
