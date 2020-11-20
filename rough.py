import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('./winequality-white.csv', sep=';')
print(df.head())

df['quality'][df['quality']!=3] = 0
df['quality'][df['quality']==3] = 1

X_train, X_test, y_train, y_test = train_test_split(df.drop(['quality'], axis=1), df['quality'], test_size=.25)

clf = SVC()
clf.fit(X_train, y_train)


# clf.intercept_ = 0