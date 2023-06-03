import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
#
# Load the breast cancer dataset
#
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

#
# Create training and test split
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

#
# Pipeline Estimator
#

pipeline = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=1))
#
# Fit the model
#

pipeline.fit(X_train, y_train)
#
# Model scores on test and training data
#

bgclassifier = BaggingClassifier(base_estimator=pipeline, n_estimators=100,
                                 max_features=10,
                                 max_samples=100,
                                 random_state=1, n_jobs=5)
#
# Fit the bagging classifier
#
bgclassifier.fit(X_train, y_train)
y_test_pred = bgclassifier.predict(X_test)

print("Recall score:", recall_score(y_test, y_test_pred))
print("Precision score:", precision_score(y_test, y_test_pred, average='weighted'))
print("Accuracy score:", accuracy_score(y_test, y_test_pred))
print("F1 score:", f1_score(y_test, y_test_pred, average='weighted'))

labels = ['M-B']
malicious = [212]
benign = [357]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, malicious, width, label='Malicious')
rects2 = ax.bar(x + width/2, benign, width, label='Benign')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of People')
ax.set_title('Grouped by malicious and benign examples')
ax.set_xticks(x, 'M-B')
ax.legend()

fig.tight_layout()

plt.show()
