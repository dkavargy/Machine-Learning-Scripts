import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import seaborn as sns

htru_2 = pd.read_csv('HTRU_2.csv')
# Let's preview the dataset
print('\n', htru_2.head())

X = htru_2.iloc[:,0:8]
Y = htru_2.iloc[:, -1]

# Check the shape of X_train and X_test
print(X.shape, Y.shape)

# Split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=11)

# Import LogisticReggresion classifier
model1 = LogisticRegression(solver='lbfgs',max_iter=1000)

# Fit classifier to training set
model1.fit(X_train, y_train)

# Make predictions on test set
y_pred = model1.predict(X_test)

# Creating roc curve

pred_prob1 = model1.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
roc_auc1 = auc(fpr1,tpr1)

print("\nMetrics before PCA:")
print("Recall score:", recall_score(y_test, y_pred,average='macro',zero_division=1))
print("Precision score:", precision_score(y_test, y_pred, average='macro',zero_division=1))
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred, average='macro'))

# Now we calculate the Logistic Regression again after we fit PCA transformation method in our model

pca = PCA(n_components=4)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Logistic Regression feature importance
model2 = LogisticRegression()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)

print("\nMetrics after PCA:")
print("Recall score:", recall_score(y_test, y_pred,average='macro',zero_division=1))
print("Precision score:", precision_score(y_test, y_pred, average='macro',zero_division=1))
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred, average='macro'))

# Calculate the most important features of our dataset
features = pd.get_dummies(X)
# Rename column names
features.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness',
              'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness']

feature_imp = pd.DataFrame({'features':list(features.columns), 'feature importance':[abs(i) for i in model1.coef_[0]]})
print('\n',feature_imp.sort_values('feature importance', ascending=False))

# Now we are going to train our model with the most important features based on model1.coef_
# The most 4 important features is: 2.IP Kurtosis, 3.IP Skewness, 5.DM-SNR, 6.DM-SNR Kurtosis

X_imp= htru_2.iloc[:,np.r_[2,3,5,6]]
Y_imp = htru_2.iloc[:, -1]

X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_imp, Y_imp, test_size=0.25,random_state=11)

model_imp = LogisticRegression()
model_imp.fit(X_train_imp,y_train_imp)

y_pred_imp = model_imp.predict(X_test_imp)

# Compute and print evaluation metrics scroe
print("\nMetrics with the 4 most important features:")
print("Recall score:", recall_score(y_test_imp, y_pred_imp,average='macro',zero_division=1))
print("Precision score:", precision_score(y_test_imp, y_pred_imp, average='macro',zero_division=1))
print("Accuracy score:", accuracy_score(y_test_imp, y_pred_imp))
print("F1 score:", f1_score(y_test_imp, y_pred_imp, average='macro'))

# Print the Confusion Matrix and slice it into four pieces

cm = confusion_matrix(y_test, y_pred_imp)

# Visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

# The confusion matrix shows 4024 + 341 = 4365 correct predictions and 23 + 87 = 110 incorrect predictions

pred_prob2 = model_imp.predict_proba(X_test_imp)
fpr2, tpr2, thresh2 = roc_curve(y_test_imp, pred_prob2[:,1], pos_label=1)
roc_auc2 = auc(fpr2, tpr2)

# Plot ROC Curve

plt.title('ROC-AUC curve')
plt.plot(fpr1, tpr1, label='ROC Curve 1 (AUC = %0.2f)' % (roc_auc1))
plt.plot(fpr2, tpr2, label='ROC Curve 2 (AUC = %0.2f)' % (roc_auc2))
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='green', label='Perfect Classifier')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")
plt.savefig('ROC-AUC curve',dpi=300)
plt.show()


