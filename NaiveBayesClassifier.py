import getopt, sys
from time import perf_counter
from math import sqrt
from math import pi
from math import exp
import numpy as np 
import pandas as pd
import statistics

n = len(sys.argv)

#for i in range(1, n):
#    print(sys.argv[i], end = " ")

file0 = sys.argv[1]
file1 = sys.argv[2]

file0AndBayes_start = perf_counter()


file0Read = pd.read_csv(file0)
file1Read = pd.read_csv(file1)
#print(file1Read[0])
#sys.stderr.print(file1Read)
#WORK ON LATER, FIRST ROW IS PAVED OVER FOR THE COLUMN LABELS
file0Read.columns = ['Hotel-Type', 'Arrival-Date-Month', 'Meal-Request', 'Market-Segment-Designation',
'Booking-Distribution-Channel', 'Reserved-Room-Type', 'Deposit-Type', 'Customer-Type', 'Days-fromReservation-to-Arrival', 
'Arrival-Date-Week-Number', 'Arrival-Date-Day-of-Month', 'Stays-inWeekend-Nights', 'Stays-in-Weekday-Nights', 'Adults', 
'Children', 'Babies', 'Is-Repeated-Guest', 'Previous-Cancellations', 'Previous-Booking-Not-Cancelled', 
'Requested-Car-Parking-Spaces', 'Total-ofSpecial-Requests', 'Average-Daily-Rate', 'Label']
#file1Read.columns = ['Hotel-Type', 'Arrival-Date-Month', 'Meal-Request', 'Market-Segment-Designation',
#'Booking-Distribution-Channel', 'Reserved-Room-Type', 'Deposit-Type', 'Customer-Type', 'Days-fromReservation-to-Arrival', 
#'Arrival-Date-Week-Number', 'Arrival-Date-Day-of-Month', 'Stays-inWeekend-Nights', 'Stays-in-Weekday-Nights', 'Adults', 
#'Children', 'Babies', 'Is-Repeated-Guest', 'Previous-Cancellations', 'Previous-Booking-Not-Cancelled', 
#'Requested-Car-Parking-Spaces', 'Total-ofSpecial-Requests', 'Average-Daily-Rate', 'Label']
file1Read.columns = ['Hotel-Type', 'Arrival-Date-Month', 'Meal-Request', 'Market-Segment-Designation',
'Booking-Distribution-Channel', 'Reserved-Room-Type', 'Deposit-Type', 'Customer-Type', 'Days-fromReservation-to-Arrival', 
'Arrival-Date-Week-Number', 'Arrival-Date-Day-of-Month', 'Stays-inWeekend-Nights', 'Stays-in-Weekday-Nights', 'Adults', 
'Children', 'Babies', 'Is-Repeated-Guest', 'Previous-Cancellations', 'Previous-Booking-Not-Cancelled', 
'Requested-Car-Parking-Spaces', 'Total-ofSpecial-Requests', 'Average-Daily-Rate', 'Label']
#print("\n",file0Read)
#print(file0Read.shape)
#print("\n-----------------------------------------------------------------------------------------------------------\n")
#print("-----------------------------------------------------------------------------------------------------------\n")
#print("-----------------------------------------------------------------------------------------------------------\n")
#print("-----------------------------------------------------------------------------------------------------------\n")
#print("-----------------------------------------------------------------------------------------------------------\n")
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
trainingClass_Y = file0Read[file0Read.columns[-1]]
#print("\n", trainingClass)

prob_features = {}
prob_features_liklihoods = {}
#This is the features columns
trainingFeatures_X = file0Read.iloc[0:file0Read.shape[0],0:22]
#print("\n", trainingFeatures)

numTrainingRows = trainingFeatures_X.shape[0]
numFeatures = trainingFeatures_X.shape[1]

#Calculate P(Label = 1) and P(Label = 0)
for result in np.unique(trainingClass_Y):
	sum_count = sum(trainingClass_Y == result)
	prob_class[result] = sum_count / numTrainingRows
#print(prob_class)

#Calculate liklihood of all discrete feature values
#print(listDiscrete)
for feature in listDiscrete:
    prob_features[feature] = {}
    for feature_value in file0Read[feature].unique():
        sum_count = sum(file0Read[feature] == feature_value)
        prob_features[feature][feature_value] = sum_count / numTrainingRows
#print(prob_features)

#Calculate yes no probabilities for each feature
#print(file0Read['feature'])
sum_count = 0
for feature in listDiscrete:
    prob_features_liklihoods[feature] = {}
    #for outcome in np.unique(trainingClass_Y):
    for feature_value in file0Read[feature].unique():
        #print("OUTCOME",outcome)
        #print("OUTCOME ", outcome, "\n")
        #for feature_value in file0Read[feature].unique():
        for outcome in np.unique(trainingClass_Y):
            #print("FEATURE_VALUE",feature_value)
            prob_features_liklihoods[feature][feature_value] = {}
            num_count = sum(file0Read[feature] == feature_value)
            #FIX LATER
            sum_count = 0
            sum_count2 = 0
            for i in range(numTrainingRows):
                if(trainingClass_Y[i] == outcome and trainingFeatures_X[feature][i] == feature_value):
                    sum_count = sum_count + 1
            sum_count2 = sum(trainingFeatures_X[feature] == feature_value)
            #sum_count = sum(trainingClass_Y == outcome)
            prob_features_liklihoods[feature][feature_value][outcome] = sum_count / sum_count2
            #prob_features_liklihoods[feature][feature_value].append({outcome : sum_count / sum_count2})
            #prob_features_liklihoods[feature][feature_value].append({1-outcome : 1-sum_count / sum_count2})
            #print("OUTCOME = ", outcome, " AND prob_features_liklihoods = ", prob_features_liklihoods)
#print(prob_features_liklihoods)

#{'Hotel-Type': {'City Hotel_1.0': [{1.0: 0.5494719807464885}, {0.0: 0.64}], 'Resort Hotel': {1.0: 0.39431687206312616}}

#Calculate the mean and std dev of the continuous features
meanD = {}
std_dev = {}
for feature in listCont:
    meanD[feature] = {}
    std_dev[feature] = {}
    for outcome in np.unique(trainingClass_Y):
        abc = file0Read[file0Read["Label"] == outcome]
        abc = abc.drop(['Hotel-Type', 'Arrival-Date-Month', 'Meal-Request', 'Market-Segment-Designation',
'Booking-Distribution-Channel', 'Reserved-Room-Type', 'Deposit-Type', 'Customer-Type', 'Is-Repeated-Guest', 'Label'], axis=1)
        #print("ABC=",abc)
        #FIX MEAN AND STD DEV
        #meanD[feature][outcome] = abc.mean()
        #std_dev[feature][outcome] = abc.std()
        res = abc[feature]
        meanD[feature][outcome] = statistics.mean(res)
        std_dev[feature][outcome] = statistics.stdev(res)
#print(meanD)
#print(std_dev)
#print("MEAN OF ADULTS = 0", meanD["Children"][1])
#print("STD OF ADULTS = 0", std_dev["Children"][0])

file0AndBayes_stop = perf_counter()
print("Elapsed time of opening the training file and training a Naive Bayes classifier: ", file0AndBayes_stop - file0AndBayes_start, "seconds")

#CALCULATING ACCURACY WITH training.txt
applyBayesOnfiles_start = perf_counter()
#print(file0Read)
#print("Accuracy of built classifier on training.txt\n")
totalAccurate = 0
totalAccurateRate = 0
rowAccuracy = 1
for ind in file0Read.index:
    #print(file0Read['Hotel-Type'][ind])
    rowAccuracy = 1
    for feature in listFeatures:
        #rowAccuracy = rowAccuracy * calculate_probability(feature, ind)
        #rowAccuracy = rowAccuracy * calculate_probability(feature, file0Read[ind])
        break
    if(rowAccuracy > 0.5):
        totalAccurate = totalAccurate + 1
totalAccurateRate = totalAccurate / numTrainingRows
print("The accuracy rate of our bayes network on the training set was ", totalAccurateRate*100, " %")


numTrainingRows2 = file1Read.shape[0]
#CALCULATING ACCURACY WITH testing.txt
#print("Accuracy of built classifier on testing.txt\n")
totalAccurate = 0
totalAccurateRate = 0
rowAccuracyYes = 1
rowAccuracyNo = 1

for ind in range(numTrainingRows2):
    #print(file0Read['Hotel-Type'][ind])
    #print("IND = ", ind)
    rowAccuracyYes = 1
    rowAccuracyNo = 1
    for feature in listFeatures:
        #print("FILE1READ.COLUMNS", file1Read.columns)
        #col = file1Read.columns.get_loc(feature)
        #print("FILE1READ", file1Read)
        feat_val = file1Read.loc[ind, feature]
        #print("FEAT_VAL", feat_val)
        #print("COLUMNS", col)
        if feature in listDiscrete:
            #Discrete
            #P(label | features) = P(feature1=city hotel | label = yes, no) * P(feature2 | label) ... * P(label)
            #P(label) = prob_class
            #P(features | label) = prob_features_liklihoods
            #print("FILE1READ[ind]=",file1Read[ind], "\nIND", ind)
            #feature_val_2 = file1Read[ind][col]
            #feature_val_2 = file1Read.iloc[ind, col]
            #print("FEATURES_VAL_2", feature_val_2)
            #print("PROB_FEATURES_LIKLIHOODS", prob_features_liklihoods)
            #print("FEATURES=", feature, "COL=", col)
            #print("prob = ",prob_features_liklihoods[feature])
            #print("prob = ",prob_features_liklihoods[feature][feat_val])
            rowAccuracyYes = rowAccuracyYes * prob_features_liklihoods[feature][feat_val][1.0]
            rowAccuracyNo = rowAccuracyNo * (1 - prob_features_liklihoods[feature][feat_val][1.0])
            #print("YES = ", rowAccuracyYes, "NO = ", rowAccuracyNo)
        if feature in listCont:
            #Continuous
            #print("con")
            #print("ZERO CHECK, 0 = ", (std_dev[feature][0.0]*std_dev[feature][0.0]), "FEATURE = ", feature)
            #print("ZERO CHECK, 1 = ", (std_dev[feature][1.0]*std_dev[feature][1.0]), "FEATURE = ", feature)
            pi = 3.14
            if((std_dev[feature][1.0]*std_dev[feature][1.0]) != 0):
                rowAccuracyYes = rowAccuracyYes * (1/sqrt(2*pi*std_dev[feature][1.0]*std_dev[feature][1.0]))*exp(-0.5 * pow((feat_val - meanD[feature][1.0]),2)/(std_dev[feature][1.0]*std_dev[feature][1.0]))
            if((std_dev[feature][0.0]*std_dev[feature][0.0]) != 0):
                rowAccuracyNo = rowAccuracyNo * (1/sqrt(2*pi*std_dev[feature][0.0]*std_dev[feature][0.0]))*exp(-0.5 * pow((feat_val - meanD[feature][0.0]),2)/(std_dev[feature][0.0]*std_dev[feature][0.0]))

            
    if (rowAccuracyNo > rowAccuracyYes):
        if file1Read.iloc[ind]["Label"] == 0:
            totalAccurate = totalAccurate + 1
    else:
        if file1Read.iloc[ind]["Label"] == 1:
            totalAccurate = totalAccurate + 1
totalAccurateRate = totalAccurate / numTrainingRows2


print("The accuracy rate of our bayes network on the testing set was ", totalAccurateRate*100, "%\n")
applyBayesOnfiles_stop = perf_counter()
print("Elapsed time of finding the accuracy of calculated bayes network on both training.txt and testing.txt: ", applyBayesOnfiles_stop - applyBayesOnfiles_start, " seconds")