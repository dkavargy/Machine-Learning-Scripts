import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import csv
from scipy import stats

friendman = pd.read_csv('algo_performance.csv')
print(friendman.head())


c4_5 = []
NN_1 = []
Naive = []
Kernel =[]
CN2 = []

with open('algo_performance.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        c4_5.append(row[0])
        NN_1.append(row[1])
        Naive.append(row[2])
        Kernel.append(row[3])
        CN2.append(row[4])

del c4_5[0]
del NN_1[0]
del Naive[0]
del Kernel[0]
del CN2[0]
#Statistical significance check is performed between  c4_5,NN_1,Naive
stat1 = stats.friedmanchisquare(c4_5,NN_1,Naive)

print(stat1)
print("p-value=0.01 > 0.05 shows as that the three different algorithms has statistical diferences")

#Statistical significance check is performed between Naive, Kernel, CN2
stat2 = stats.friedmanchisquare(Naive,Kernel,CN2)
print(stat2)
print("p-value=7.2790625505492486e-06< 0.05 shows as that the three different algorithms has not statistical diferences")