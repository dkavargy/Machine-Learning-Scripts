# =============================================================================
# HOMEWORK 2 - DECISION TREES
# RANDOM FOREST ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'ensemble' package, for calling the Random Forest classifier
# 'model_selection', (instead of the 'cross_validation' package), which will help validate our results.
# =============================================================================


# IMPORT NECESSARY LIBRARIES HERE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# Load breastCancer data
# =============================================================================

# ADD COMMAND TO LOAD DATA HERE
breastCancer = load_breast_cancer()

# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 9
X = breastCancer.data[:, 0:numberOfFeatures]
y = breastCancer.target

# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Also, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=0)

# RandomForestClassifier() is the core of this script. You can call it from the 'ensemble' class.
# You can customize its functionality in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the Information Gain.
# 'n_estimators': The number of trees in the forest. The larger the better, but it will take longer to compute. Also,
#                 there is a critical number after which there is no significant improvement in the results
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================

# ADD COMMAND TO CREATE RANDOM FOREST CLASSIFIER MODEL HERE
RandomForestModel = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=3)

# Let's train our model.
# =============================================================================

# ADD COMMAND TO TRAIN YOUR MODEL HERE

RandomForestModel.fit(x_train,y_train)

# Ok, now let's predict the output for the test set
# =============================================================================

# ADD COMMAND TO MAKE A PREDICTION HERE
y_predicted = RandomForestModel.predict(x_test)

# Time to measure scores. We will compare predicted output (from input of second subset, i.e. x_test)
# with the real output (output of second subset, i.e. y_test).
# You can call 'accuracy_score', 'recall_score', 'precision_score', 'f1_score' or any other available metric
# from the 'sklearn.metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# One of the following can be used for this example, but it is recommended that 'macro' is used (for now):
# 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
#             This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
# =============================================================================

# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("Recall score:", recall_score(y_test, y_predicted))
print("Precision score:", precision_score(y_test, y_predicted, average='weighted'))
print("Accuracy score:", accuracy_score(y_test, y_predicted))
print("F1 score:", f1_score(y_test, y_predicted, average='weighted'))

# A Random Forest has been trained now, but let's train more models,
# with different number of estimators each, and plot performance in terms of
# the difference metrics. In other words, we need to make 'n'(e.g. 200) models,
# evaluate them on the aforementioned metrics, and plot 4 performance figures
# (one for each metric).
# In essence, the same pipeline as previously will be followed.
# =============================================================================

# CREATE MODELS AND PLOTS HERE

# Calculate aforementioned metrics for all the possible estimators from 1 to 200.

n_estimators = 200
batch_size = [] * 200
accuracy = []
recall = []
precision = []
f1 = []
for i in range(n_estimators):
    #print("Number of estimators:", i+1)
    batch_size.append(i+1)
    RandomForestModel = RandomForestClassifier(n_estimators=i+1, criterion='gini', max_depth=3)
    RandomForestModel.fit(x_train, y_train)
    y_predicted = RandomForestModel.predict(x_test)
    n_estimator_accuracy = accuracy_score(y_test, y_predicted)
    n_estimator_recall = recall_score(y_test, y_predicted)
    n_estimator_precision =  precision_score(y_test, y_predicted)
    n_estimator_f1 = f1_score(y_test, y_predicted)
    accuracy.append(n_estimator_accuracy)
    recall.append(n_estimator_recall)
    precision.append(n_estimator_precision)
    f1.append(n_estimator_f1)
   # print("Accuracy score:", n_estimator_accuracy)

#Remove the comments from the following commands to see the plots for each aforementioned metric
#For convenience, only the f1 score plot has been kept

#plt.plot(batch_size,accuracy,'b-o',label='Accuracy for 200 estimators')
#plt.plot(batch_size,recall,'tab:orange', label='Recall for 200 estimators')
#plt.plot(batch_size,precision,'tab:red',label='Precision for 200 estimators')
plt.plot(batch_size,f1,'tab:grey',label='f1 score for 200 estimators')
plt.axis([0, 200, 0.9, 1])
plt.xlabel('Number of estimators')
plt.ylabel('F1 score')
plt.legend()
plt.show()

