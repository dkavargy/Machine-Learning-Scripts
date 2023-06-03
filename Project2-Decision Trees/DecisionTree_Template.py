# =============================================================================
# HOMEWORK 2 - DECISION TREES
# DECISION TREE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# IMPORT NECESSARY LIBRARIES HERE
# classifying
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn import tree

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load breastCancer data
# =============================================================================

# ADD COMMAND TO LOAD DATA HERE
breastCancer = load_breast_cancer()
# =============================================================================

# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
# For the classifier we will only used the first 10 (those are the ones that contain the average values).
numberOfFeatures = 9
X = breastCancer.data[:, 0:numberOfFeatures]
y = breastCancer.target

# DecisionTreeClassifier() is the core of this script. You can customize its functionality
# in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the information gain.
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================


# ADD COMMAND TO CREATE DECISION TREE CLASSIFIER MODEL HERE
clf_dt = DecisionTreeClassifier(criterion="gini", max_depth=3)

# =============================================================================

# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# ADD COMMAND TO TRAIN YOUR MODEL HERE
clf_dt.fit(x_train,y_train)

# Ok, now let's predict the output for the test input set
# =============================================================================

# ADD COMMAND TO MAKE A PREDICTION HERE
y_predicted = clf_dt.predict(x_test)

# =============================================================================

# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'recall_score()', 'precision_score()', 'accuracy_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# =============================================================================

# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("Recall score:", recall_score(y_test, y_predicted))
print("Precision score:", precision_score(y_test, y_predicted, average='weighted'))
print("Accuracy score:", accuracy_score(y_test, y_predicted))
print("F1 score:", f1_score(y_test, y_predicted, average='weighted'))

# =============================================================================

# By using the 'plot_tree' function from the tree classifier we can visualize the trained model.
# There is a variety of parameters to configure, which can lead to a quite visually pleasant result.
# Make sure that you set the following parameters within the function:
# feature_names = breastCancer.feature_names[:numberOfFeatures]
# class_names = breastCancer.target_names
# filled = True
# =============================================================================

plt.figure(figsize=(10, 5))
tree.plot_tree(clf_dt,class_names=True, rounded=True)

plt.show()

