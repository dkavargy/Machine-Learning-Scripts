# =============================================================================
# HOMEWORK 5 - BAYESIAN LEARNING
# NAIVE BAYES ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# From sklearn, we will import:
# 'datasets', for loading data
# 'model_selection' package, which will help validate our results
# 'metrics' package, for measuring scores
# 'naive_bayes' package, for creating and using Naive Bayes classfier
from sklearn import  model_selection
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
# We also need to import 'make_pipeline' from the 'pipeline' module.

# We are working with text, so we need
# an appropriate package
# that shall vectorize words within our texts.
# 'TfidfVectorizer' from 'feature_extraction.text'.
from sklearn.feature_extraction.text import TfidfVectorizer

# 'matplotlib.pyplot' and 'seaborn' are ncessary as well,
# for plotting the confusion matrix.
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load text data.
categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']


newsgroups_train = fetch_20newsgroups(subset="train", categories=categories,remove=("headers", "footers", "quotes"))
newsgroups_test = fetch_20newsgroups(subset="test", categories=categories)

X = newsgroups_train.data

y = newsgroups_train.target

# Store features and target variable into 'X' and 'y'.

# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 0, test_size=0.33)

# We need to perform a transformation on the model that will later become
# our Naive Bayes classifier. This transformation is text vectorization,
# using TfidfVectorizer().
# When you want to apply several transformations on a model, and an
# estimator at the end, you can use a 'pipeline'. This allows you to
# define a chain of transformations on your model, like a workflow.
# In this case, we have one transformer that we wish to apply (TfidfVectorizer)
# and an estimator afterwards (Multinomial Naive Bayes classifier).
# =============================================================================

# ADD COMMAND TO MAKE PIPELINE HERE
alpha = 0.1 # This is the smoothing parameter for Laplace/Lidstone smoothing

# Let's train our model.
# =============================================================================

# ADD COMMAND TO TRAIN MODEL HERE
model = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=alpha))
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# =============================================================================

# Ok, now let's predict output for the second subset
# =============================================================================


# ADD COMMAND TO MAKE PREDICTION HERE


# =============================================================================


# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'accuracy_score()', recall_score()', 'precision_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform
# a type of averaging on the data. Use 'macro' for final results.
# =============================================================================


# ADD COMMANDS TO COMPUTE METRICS HERE (AND PRINT ON CONSOLE)

print("Recall score:", recall_score(y_test, y_pred,average='macro'))
print("Precision score:", precision_score(y_test, y_pred, average='macro'))
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred, average='macro'))


# In order to plot the 'confusion_matrix', first grab it from the 'metrics' module
# and then throw it within the 'heatmap' method from the 'seaborn' module.
# =============================================================================

# ADD COMMANDS TO PLOT CONFUSION MATRIX

labels_x = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

labels_y=['atheism',
 'graphics',
 'ms-windows.misc',
 'sys.ibm.pc.hardware',
 'sys.mac.hardware',
 'windows.x',
 'forsale',
 'autos',
 'motorcycles',
 'sport.baseball',
 'sport.hockey',
 'crypt',
 'electronics',
 'med',
 'space',
 'religion.christian',
 'politics.guns',
 'politics.mideast',
 'politics.misc',
 'religion.misc']
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, linewidths=1, annot=True, fmt='g',xticklabels=labels_y,yticklabels=labels_x,cmap="YlGnBu")
#print(cf_matrix)
plt.title('Multinomial NB - Confusion Matrix(a=0.10)(Recall=0.71, Precision=0.77, Accuracy=0.73, F1=0.71)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

