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

melbourne = pd.read_csv('camera_dataset.csv')
#print(melbourne.to_string())
#melbourne['Bedroom2'] = melbourne['Bedroom2'].fillna(method='ffill')
#melbourne['Bathroom'] = melbourne['Bathroom'].fillna(method='ffill')
print(melbourne.isna().sum())
print('DataFrame after dropping the rows having missing values:')

X = melbourne.iloc[:, np.r_[3,4,5,6]]
Y = melbourne.iloc[:, 7]
print(X.shape, Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=1)
cv = LeaveOneOut()
model = RandomForestClassifier(random_state=1,n_estimators=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy', acc)
scores = cross_val_score(model, X, Y,  cv=cv)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
cm = confusion_matrix(y_test, y_pred)
print('True positive = ', cm[0][0])
print('False positive = ', cm[0][1])
print('False negative = ', cm[1][0])
print('True negative = ', cm[1][1])
print(cm)
display_labels=['Survivde', 'Dided']
#disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=display_labels)
#ax=sns.heatmap(cm,annot=True,cmap='Blues',fmt='d')
plt.show()



