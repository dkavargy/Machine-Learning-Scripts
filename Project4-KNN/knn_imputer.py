# =============================================================================
# HOMEWORK 4 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email: arislaza@csd.auth.gr
# =============================================================================

# import the KNeighborsClassifier
# if you want to do the hard task, also import the KNNImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# Import the titanic dataset
# Decide which features you want to use (some of them are useless, ie PassengerId).
#
# Feature 'Sex': because this is categorical instead of numerical, KNN can't deal with it, so drop it
# Note: another solution is to use one-hot-encoding, but it's out of the scope for this exercise.
#
# Feature 'Age': because this column contains missing values, KNN can't deal with it, so drop it
# If you want to do the harder task, don't drop it.
#
# =============================================================================
titanic = pd.read_csv('titanic.csv')
titanic.drop(['Sex','Cabin','Name','Ticket'], inplace=True,axis=1)

cat_variables = titanic['Embarked']
cat_dummies = pd.get_dummies(cat_variables, drop_first=True)


titanic=titanic.drop(['Embarked'],axis=1)
titanic = pd.concat([titanic, cat_dummies], axis=1)


#col_mapping_dict = {c[0]:c[1] for c in enumerate(titanic.columns)}


# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================
# split into train test sets
#train, test = train_test_split(titanic)

X = titanic.iloc[:, np.r_[1:3,4,5,7]]
y = titanic.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

scaler = MinMaxScaler()
titanic=pd.DataFrame(scaler.fit_transform(titanic), columns=titanic.columns)

n_neighbors = 200
batch_size = [] * 200
f1 = []
for i in range(n_neighbors):
    batch_size.append(i+1)
    classifier = KNeighborsClassifier(n_neighbors=i+1, weights='uniform', p=2, metric='minkowski')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    n_f1 = f1_score(y_test,y_pred, average='macro')
    #print(n_f1)
    f1.append(n_f1)

print(titanic.isna().sum())
imputer = KNNImputer(n_neighbors=3)
#titanic=pd.DataFrame(imputer.fit_transform(titanic), columns=titanic.columns)
titanic['Age'] = titanic['Age'].fillna(method='ffill')
#titanic=pd.DataFrame(scaler.fit_transform(titanic), columns=titanic.columns)

print(titanic.isna().sum())

f1_impute = []
best_f1=0
for i in range(n_neighbors):
    classifier = KNeighborsClassifier(n_neighbors=i+1, weights='uniform', p=2, metric='minkowski')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    n_f1_impute = f1_score(y_test,y_pred, average='macro')
    #print(n_f1_impute)
    f1_impute.append(n_f1_impute)


print("Recall score:", recall_score(y_test, y_pred,average='macro',zero_division=1))
print("Precision score:", precision_score(y_test, y_pred, average='macro',zero_division=1))
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred, average='macro'))

plt.title('k-Nearest Neighbors (Weights = uniform, Metric = mink, p = 2)')
plt.plot(batch_size,f1,'tab:grey',label='f1 score for n=40(without impute)')
plt.plot(batch_size,f1_impute,'tab:red',label='f1 score for n=40(with impute)')
plt.axis([0, 200, 0.2, 1])
plt.xlabel('Number of neighbors')
plt.ylabel('F1 score')
plt.legend()
plt.show()

'''

# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer.
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================
imputer = KNNImputer(n_neighbors=3, weights='uniform', metric='nan_euclidean')
imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================

classifier = KNeighborsClassifier(n_neighbors=20, weights='uniform', p=1, metric='minkowski')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Recall score:", recall_score(y_test, y_pred,average='macro',zero_division=1))
print("Precision score:", precision_score(y_test, y_pred, average='macro',zero_division=1))
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred, average='macro'))

# Plot the F1 performance results for any combination οf parameter values of your choice.
# If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)

# =============================================================================
'''


