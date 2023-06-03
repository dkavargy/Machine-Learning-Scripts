from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.axes as ax
from numpy import mean
from numpy import std

#loading vaw dataset
df = pd.read_csv('VAW.csv')


df.drop(['FREQ: Frequency', 'SEX: Sex', 'UNIT_MEASURE: Unit of measure', 'OBS_STATUS: Observation Status', 'OBS_VALUE', 'UNIT_MULT: Unit multiplier', 'OBS_COMMENT: Comment'], axis=1, inplace=True)
df.loc[df['OUTCOME: Outcome'] == 'INJ: Injured', 'INJ'] = 1
print(df)
print(df.isnull().sum())
#X = df.iloc[:, np.r_[0:10, 12:]]
X = df.loc[:,df.columns != "OUTCOME: Outcome"]
Y = df['OUTCOME: Outcome']
Y.replace('_T: Any','missing_data', inplace=True)

Y.replace('INJ: Injured', 1, inplace=True)
#Y.replace('_T: Any', 0, inplace=True)

#Y.drop(Y.index[Y == 'missing_data'], inplace=True)
#Y = df.iloc[:, 11]
#new_Y = Y.dropna( axis=0, how='any', inplace=False)
#new_new_Y = df.drop(df.loc[df['OUTCOME: Outcome'] == '_T: Any'].index,inplace=True)
print('The nEW Y is:', Y)
print('The x is:', Y.shape, X.shape)
print(df)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)


count = y_train.value_counts()
count.plot.bar()
plt.ylabel('Number of records')
plt.xlabel('Target Class')
plt.show()