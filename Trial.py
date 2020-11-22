import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imbalacedSVM import ImbalancedSVC

def WineQualityWhite():
    df = pd.read_csv('./winequality-white.csv', sep=';')

    # 3-6: Bad(0), 7-10: Good(1)
    df['quality'] = df['quality'].apply(lambda x: int(x<7))
    # print(df.describe())

    y = df.pop("quality")
    #smote = SMOTE()
    #X, y = smote.fit_resample(df, y)
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

    clf = SVC(kernel='linear')
    # clf = ImbalancedSVC(kernel='linear')
    print(1)
    clf.fit(X_train, y_train)
    print(2)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    #print("Accuracy for White Wine: ", )

def WineQualityRed():
    df = pd.read_csv('./winequality-red.csv', sep=';')

    # 3-6: Bad(0), 7-10: Good(1)
    df['quality'] = df['quality'].apply(lambda x: int(x<7))
    # print(df.describe())

    y = df.pop("quality")

    smote = SMOTE()
    X, y = smote.fit_resample(df, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

    # clf = SVC(class_weight={0:0.15, 1:0.85})
    # clf = SVC(kernel='linear')
    clf = ImbalancedSVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # print("Accuracy for Red Wine: ", clf.score(X_test, y_test))

# print("\nWhite Wine")
# WineQualityWhite()
print("Red Wine")
WineQualityRed()