from sklearn.model_selection import LeaveOneOut, cross_val_score
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
#Read the dataset that contains melbourne data
melbourne = pd.read_csv('melb_data.csv')
#Drop the uneccesary columns
melbourne.drop(['Address', 'BuildingArea','Postcode'],inplace=True,axis=1)

#print(melbourne.to_string())
#fill the NaN Values in the below variables
melbourne['Bedroom2'] = melbourne['Bedroom2'].fillna(method='ffill')
melbourne['Bathroom'] = melbourne['Bathroom'].fillna(method='ffill')
print(melbourne.isna().sum())

X = melbourne.iloc[:, np.r_[2,4,9,10]]
Y = melbourne.iloc[:, 3]
print(X.shape, Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

#Leave One Out method
cv = LeaveOneOut()

#I Choose Random Forest classifier to train my model
model = RandomForestClassifier(random_state=1,n_estimators=10)

#Fit and predict my model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy', acc)
#Calulating the cross validation score. However, the dataset i utilize is very big so the executing time is very big.
#Remove the commects to see the real results

'''
scores = cross_val_score(model, X, Y,  cv=cv)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
'''

#Calculating True Positive, True Neagtive etc
cm = confusion_matrix(y_test, y_pred)
print('True positive = ', cm[0][0])
print('False positive = ', cm[0][1])
print('False negative = ', cm[1][0])
print('True negative = ', cm[1][1])
print(cm)

#Plot the confusion matrix
ax=sns.heatmap(cm,annot=True,cmap='Blues',fmt='d')

plt.title('Confusion Matrix')
plt.show()


