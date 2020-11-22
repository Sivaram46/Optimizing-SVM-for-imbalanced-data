import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold


def WineQualityWhite():
    df = pd.read_csv('./winequality-white.csv', sep=';')

    # 3-6: Bad(0), 7-10: Good(1)
    df['quality'] = df['quality'].apply(lambda x: int(x<7))
    # print(df.describe())

    y = df.pop("quality")
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.25)

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    #print("Accuracy for White Wine: ", )

def WineQualityRed():
    df = pd.read_csv('./winequality-red.csv', sep=';')

    # 3-6: Bad(0), 7-10: Good(1)
    df['quality'] = df['quality'].apply(lambda x: int(x<7))
    # print(df.describe())

    y = df.pop("quality")
    #X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.25)

    skf = StratifiedKFold(n_splits=2)
    StratifiedKFold(n_splits=2, shuffle=True)
    for train_index, test_index in skf.split(df, y):
        X_train, X_test = df.iloc[train_index].values, df.iloc[test_index].values
        y_train, y_test = y[train_index].values, y[test_index].values

    print(y_train.mean(), y_test.mean())

    # clf = SVC(class_weight={0:0.15, 1:0.85})
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # print("Accuracy for Red Wine: ", clf.score(X_test, y_test))

WineQualityRed()
# WineQualityWhite()