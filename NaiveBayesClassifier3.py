import getopt, sys
from time import perf_counter
from math import sqrt
from math import pi
from math import exp
import numpy as np 
import pandas as pd





n = len(sys.argv)

for i in range(1, n):
    print(sys.argv[i], end = " ")

file0 = sys.argv[0]
file1 = sys.argv[1]

file0AndBayes_start = perf_counter()

file0Read = pd.read_csv(file0)
file1Read = pd.read_csv(file1)

X = file0Read.drop([file0Read.columns[-1]], axis = 1)
y = file0Read[file0Read.columns[-1]]

print("X IS: ", X)
print("Y IS: ", y)

listFeatures = []
trainLikelihoods = {}
trainClass = {}
trainFeatures = {}


file0AndBayes_stop = perf_counter()
print("Elapsed time of opening the training file and training a Naive Bayes classifier: ", file0AndBayes_stop - file0AndBayes_start)


applyBayesOnfile1_start = perf_counter()

applyBayesOnfile1_stop = perf_counter()







