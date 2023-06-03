from scipy.stats import stats
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, model_selection

# Load diabetes data from 'datasets' class
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Load just 1 feature for simplicity and visualization purposes...
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets

x_train, x_test, y_train, y_test = model_selection.train_test_split(diabetes_X, diabetes_y, test_size = 0.33, random_state = 0)

# Create linear regression object
linearReggresionModel = linear_model.LinearRegression()

# Train the model using the training sets
linearReggresionModel.fit(x_train, y_train)

# Predict the output for the test input set
y_predicted = linearReggresionModel.predict(x_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_predicted))
print("--------Corellation----------:")
print('#', stats.spearmanr(y_test, y_predicted))
print('# PearsonrResult:', stats.pearsonr(y_test, y_predicted))

# Plot results in a 2D plot (scatter() plot, line plot())
# Display 'ticks' in x-axis and y-axis
#Create a scatterplot of the real test values versus the predicted values.

plt.scatter(x_test, y_test, color="black")
plt.plot(x_test, y_predicted, color="blue", linewidth=3)
plt.xlabel('y Test')
plt.ylabel('Predicted y')
plt.xticks(())
plt.yticks(())

plt.show()