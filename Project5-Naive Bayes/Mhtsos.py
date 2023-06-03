from random import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from matplotlib import pyplot as plt
import numpy as np
from numpy import mean, std


# summarize the sonar dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)

cv = LeaveOneOut()

model = RandomForestClassifier(random_state=1)

scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


print(mean(scores))


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


model = RandomForestClassifier(n_estimators=10,criterion="entropy",max_depth=3)


model.fit(x_train,y_train)


y_predicted = model.predict(x_test)



# print(accuracy_score(y_test,y_predicted))
# print(precision_score(y_test,y_predicted,average='macro'))
# print(recall_score(y_test,y_predicted,average="macro"))
# print(f1_score(y_test,y_predicted,average='macro'))