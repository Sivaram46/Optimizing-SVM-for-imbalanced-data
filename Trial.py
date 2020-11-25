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
from sklearn.ensemble import RandomForestClassifier

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

    X = df.values

    #smote = SMOTE()
    #X, y = smote.fit_resample(df, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

    # clf = SVC(class_weight={0:0.15, 1:0.85})
    # clf = SVC(kernel='linear')
    # clf = ImbalancedSVC(kernel='linear')
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # print(clf.decision_function(X_test))
    # print("Accuracy for Red Wine: ", clf.score(X_test, y_test))

def Ensemble():

    df = pd.read_csv('./winequality-red.csv', sep=';')

    # 3-6: Bad(1), 7-10: Good(0)
    df['quality'] = df['quality'].apply(lambda x: int(x<7))
    y = df.pop("quality")
    X = df.values

    # Splitting the Real Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Splitting the Training data into Training and Testing Again
    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=0.30)

    rf = RandomForestClassifier(n_estimators=80, n_jobs=-1)
    rf.fit(X_train_train, y_train_train)
    y_pred_probs = rf.predict_proba(X_train_test)
    # Indices of those RF not very confident in predicting
    # We have a threshold of 0.15 here 
    indices = (np.where(np.logical_and(y_pred_probs>=0.30, y_pred_probs<=0.70))[0])


    # Use SVM when RF not confident
    svm = SVC(kernel='linear')
    svm.fit(X_train_test[indices], y_train_test.values[indices])
    
    y_pred = []
    for data in X_test:
        # Making a 1D Array as 2D array since sklearn expects a 2D array for fitting and predicting
        data = [data]
        # Return the prob for each class but as a 2d list
        proba = rf.predict_proba(data)
        # Taking only one, since we take only a specific range
        prob = proba.flatten()[0] # prob is a list with prob for each class
        
        # If RF not confident in predicting the class, use SVM
        if(prob>=0.30 and prob<=0.70):
            prediction = svm.predict(data)
            y_pred.append(prediction)
        # If RF confident, go with the prediction
        else:
            prediction = rf.predict(data)
            y_pred.append(prediction)

    print(classification_report(y_test, y_pred))

# print("\nWhite Wine")
# WineQualityWhite()

# print("\nRed Wine")
# WineQualityRed()

# print("\nEnsemble")
# Ensemble()

