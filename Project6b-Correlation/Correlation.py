import csv
from scipy import stats
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn import preprocessing
from matplotlib import pyplot
import numpy as np
from sklearn.preprocessing import MinMaxScaler

correlation = pd.read_csv('GeorgeData.csv', sep=';')
correlation = correlation[['Step', 'Value']]
#Get values from the two different columns of the DataFrame


#MinMaxScaler is a class from sklearn.preprocessing which is used for normalization.
mmscaler = MinMaxScaler()
cols = ['Step', 'Value']
correlation[cols] = mmscaler.fit_transform(correlation[cols])

step = correlation.loc[:, 'Step']
value = correlation.loc[:, 'Value']


#Now we can use the normalize() method on the array. This method normalizes data along a row.

#We can calculate the correlation between the two variables in our test problem with Pearson’s correlation coefficient.
pearson_corr, _ = stats.pearsonr(step,value)
print('Pearsonr correlation:', pearson_corr)

spearmanr_corr, _ = stats.spearmanr(step, value)
print('Spearmanr correlation: ',spearmanr_corr)

#We know that the data is Gaussian and that the relationship between the variables is linear.
#Nevertheless, the nonparametric rank-based approach shows a small correlation between the variables of 0.2355.

#A scatter plot of the two variables is created.
#Because we contrived the dataset, we know there is not a relationship between the two variables.

pyplot.scatter(step,value)
pyplot.show()
