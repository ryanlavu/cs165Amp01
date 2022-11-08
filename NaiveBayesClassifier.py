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

file0 = sys.argv[1]
file1 = sys.argv[2]

file0AndBayes_start = perf_counter()

file0Read = pd.read_csv(file0)
file1Read = pd.read_csv(file1)

#WORK ON LATER, FIRST ROW IS PAVED OVER FOR THE COLUMN LABELS
file0Read.columns = ['Hotel-Type', 'Arrival-Date-Month', 'Meal-Request', 'Market-Segment-Designation',
'Booking-Distribution-Channel', 'Reserved-Room-Type', 'Deposit-Type', 'Customer-Type', 'Days-fromReservation-to-Arrival', 
'Arrival-Date-Week-Number', 'Arrival-Date-Day-of-Month', 'Stays-inWeekend-Nights', 'Stays-in-Weekday-Nights', 'Adults', 
'Children', 'Babies', 'Is-Repeated-Guest', 'Previous-Cancellations', 'Previous-Booking-Not-Cancelled', 
'Requested-Car-Parking-Spaces', 'Total-ofSpecial-Requests', 'Average-Daily-Rate', 'Label']
#print("\n",file0Read)
#print(file0Read.shape)
print("\n-----------------------------------------------------------------------------------------------------------\n")
print("-----------------------------------------------------------------------------------------------------------\n")
print("-----------------------------------------------------------------------------------------------------------\n")
print("-----------------------------------------------------------------------------------------------------------\n")
print("-----------------------------------------------------------------------------------------------------------\n")
#file0Read.info()

#Want to preprocess to now
listFeatures = ['Hotel-Type', 'Arrival-Date-Month', 'Meal-Request', 'Market-Segment-Designation',
'Booking-Distribution-Channel', 'Reserved-Room-Type', 'Deposit-Type', 'Customer-Type', 'Days-fromReservation-to-Arrival', 
'Arrival-Date-Week-Number', 'Arrival-Date-Day-of-Month', 'Stays-inWeekend-Nights', 'Stays-in-Weekday-Nights', 'Adults', 
'Children', 'Babies', 'Is-Repeated-Guest', 'Previous-Cancellations', 'Previous-Booking-Not-Cancelled', 
'Requested-Car-Parking-Spaces', 'Total-ofSpecial-Requests', 'Average-Daily-Rate']
#listDiscrete = list(range(0,8))
#listDiscrete.append(17)
listDiscrete = ['Hotel-Type', 'Arrival-Date-Month', 'Meal-Request', 'Market-Segment-Designation',
'Booking-Distribution-Channel', 'Reserved-Room-Type', 'Deposit-Type', 'Customer-Type', 'Is-Repeated-Guest']
#listCont = list(range(8,17))+list(range(18,23))
listCont = ['Days-fromReservation-to-Arrival', 'Arrival-Date-Week-Number', 'Arrival-Date-Day-of-Month', 'Stays-inWeekend-Nights', 'Stays-in-Weekday-Nights', 'Adults', 
'Children', 'Babies','Previous-Cancellations', 'Previous-Booking-Not-Cancelled', 'Requested-Car-Parking-Spaces', 'Total-ofSpecial-Requests', 'Average-Daily-Rate']
trainingLikelihoods = {}

prob_class = {}
#This is the class column
trainingClass = file0Read[file0Read.columns[-1]]
#print("\n", trainingClass)

prob_features = {}
#This is the features columns
trainingFeatures = file0Read.iloc[0:file0Read.shape[0],0:22]
#print("\n", trainingFeatures)

numTrainingRows = trainingFeatures.shape[0]
numFeatures = trainingFeatures.shape[1]

#Calculate P(Label = 1) and P(Label = 0)
for result in np.unique(trainingClass):
		sum_count = sum(trainingClass == result)
		prob_class[result] = sum_count / numTrainingRows
#print(prob_class)

#Calculate liklihood of all discrete feature values
#print(listDiscrete)
for feature in listDiscrete:
    prob_features[feature] = {}
    for feature_value in file0Read[feature].unique():
        sum_count = sum(file0Read[feature] == feature_value)
        prob_features[feature][feature_value] = sum_count / numTrainingRows
print(prob_features)

#Calculate 
#for feature in listDiscrete:
#    prob_features[feature] = {}
#    for outcome in np.unique(trainingClass):
#        for feature_value in file0Read[feature].unique():
#            sum_count = sum(trainingClass == outcome and i for i in )
#            prob_features[feature][outcome] = sum_count / numTrainingRows


file0AndBayes_stop = perf_counter()
print("Elapsed time of opening the training file and training a Naive Bayes classifier: ", file0AndBayes_stop - file0AndBayes_start)


applyBayesOnfile1_start = perf_counter()

applyBayesOnfile1_stop = perf_counter()