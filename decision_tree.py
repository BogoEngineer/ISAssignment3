import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rand
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


def main():
    # pd.set_option('display.max_columns', 15)
    # pd.set_option('display.width', None)
    data = pd.read_csv("cakes_train.csv")

    print(data.head())
    print(data.tail())

    print(data.info())
    print(data.describe())

    # plotting correlation of y and each individual feature
    plt.ylabel("Type")
    for feature in data:
        if feature == "type": continue
        fig = plt.gcf()
        fig.canvas.set_window_title(feature + " correlation to cake type")
        plt.xlabel(feature)
        plt.scatter(data[feature], data.type)
        plt.show()


    # correlation matrix plot
    names = ['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder', 'type']
    correlations = data.corr()
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 7, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()

    # normalization
    for feature in data:
        if feature == "type": continue
        data[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())

    # classifier
    dtc = DecisionTreeClassifier(criterion="entropy")

    split_row = int(data.shape[0] * 0.7)
    train, test = data.iloc[:split_row, :], data.iloc[split_row:, :]

    X = train[['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']]
    Y = train['type']
    dtc.fit(X, Y)

    X = test[['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']]
    Y = test['type']

    score = dtc.score(X, Y)

    print("SKLEARN SCORE: ", score)

    plot_tree(dtc)

if __name__ == "__main__":
    main()
