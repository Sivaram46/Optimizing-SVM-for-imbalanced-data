import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def WineQualityWhite():
    df = pd.read_csv('./winequality-white.csv', sep=';')

    # 3-6: Bad(0), 7-10: Good(1)
    df['quality'] = df['quality'].apply(lambda x: int(x<7))
    #print(df.head())

    y = df.pop("quality")
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.25)

    clf = SVC()
    clf.fit(X_train, y_train)
    print("Accuracy for White Wine: ", clf.score(X_test, y_test))

def WineQualityRed():
    df = pd.read_csv('./winequality-red.csv', sep=';')

    # 3-6: Bad(0), 7-10: Good(1)
    df['quality'] = df['quality'].apply(lambda x: int(x<7))
    #print(df.head())

    y = df.pop("quality")
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.25)

    clf = SVC()
    clf.fit(X_train, y_train)
    print("Accuracy for Red Wine: ", clf.score(X_test, y_test))

WineQualityRed()
WineQualityWhite()