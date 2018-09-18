# /* A KNN implementation without Scikit learn
#  * 
#  * Input parameters:inputData.csv
#  *      
#  * Execute file: place inputData.csv and glaccount.py in the same
#  * directory
#  *
#  * Filename: <glaccount.py>
#  * Author  : <Lloyd .M. Dzokoto>
#  * Date    : <11.09.2018>
#  * Version : <1>
#  *
#  */

import csv
import pandas as pd
import numpy as np
import random
import math
import operator
from collections import Counter


glaccounts = [34010100, 62220255, 70500350, 73030016, 73620900, 73710210]

def getData(filename):
	data = pd.read_csv(filename)

	data = data.astype(float).values.tolist()
	random.shuffle(data)

	return data


def dataSplit(data,testTargets, testFeatures, trainTargests, trainFeatures,test_size):

	
	testing_data = data[:-int(test_size*len(data))]
	training_data = data[-int(test_size*len(data)):]

	#split testing data into features and targets
	for i in range(len(testing_data)):
		testTargets.append(testing_data[i][-1])

	for u in range(len(testing_data)):
		testFeatures.append(testing_data[u][:-1])


	#split training data into features and targets
	for t in range(len(training_data)):
		trainTargests.append(training_data[t][-1])

	for x in range(len(training_data)):
		trainFeatures.append(training_data[x][:-1])

	return testing_data, training_data


def distance(instance1, instance2):
    instance1 = np.array(instance1) 
    instance2 = np.array(instance2)

    # distance measurement between two instances
    return np.linalg.norm(instance1 - instance2)

# calculating the distance from new instance to all other data instances
def get_neighbors(training_set, labels, test_instance, k, distance=distance):
    distances = []
    kneighbors = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
        kneighbors.append((labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]

    return neighbors


def majorityVote(neighbors,):
    class_counter = Counter()
    for neighbor in neighbors:
    	class_counter[neighbor[2]] += 1
    return class_counter.most_common(1)[0][0]


def probabilityVote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
    	class_counter[neighbor[2]] += 1
    labels, votes = zip(*class_counter.most_common())
    predictedWinner = class_counter.most_common(2)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    return predictedWinner, votes4winner/sum(votes)


def weightedVote(neighbors, results=True):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
    	class_counter[neighbors[index][2]] += 1/(index+1)
    labels, votes = zip(*class_counter.most_common())
    predictedWinner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if results:
        total = sum(class_counter.values(), 0.0)
        for index in class_counter:
        	class_counter[index] /= total
        return predictedWinner, class_counter.most_common()
    else:
        return predictedWinner, votes4winner / sum(votes)

def weightedDistance(neighbors, results=True):
    class_counter = Counter()
    kneighbors = len(neighbors)
    for index in range(kneighbors):
    	dist = neighbors[index][1]
    	label = neighbors[index][2]
    	class_counter[label] += 1 / (dist**2 + 1)
    labels, votes = zip(*class_counter.most_common())
    predictedWinner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    
    if results:
    	total = sum(class_counter.values(), 0.0)
    	for key in class_counter:
    		class_counter[key] /= total
    	return predictedWinner, class_counter.most_common()
    else:
        return predictedWinner, votes4winner / sum(votes)

def main():

    non_WeightedPrediction = []
    WeightedPrediction = []
    kneighbors = []
    glLabels = []
    dataSet = getData('fdata3.csv')
    testTargets =   []
    testFeatures =  []
    trainTargests = []
    trainFeatures = []
    splitRatio = 0.67
    testData, trainData = dataSplit(dataSet, testTargets, testFeatures, trainTargests, trainFeatures, splitRatio)
    print('Train set: ' + repr(len(trainData)))
    print('Test set: ' + repr(len(testData)))


    #trial_data = [[3,39], [2,75], [3,59]]
    #trial_labels = [34010100]

    for i in range(len(testTargets)):
        neighbors = get_neighbors(trainFeatures, trainTargests, testFeatures[i],3,distance=distance)

        vote_prob1 = probabilityVote(neighbors)[0]
        vote_prob2 = probabilityVote(neighbors)[1]

        temp =  weightedDistance(neighbors,results=True)
        print("index: ", i)
        print('predicted = ', glaccounts[int(vote_prob1)], ',  GL account class = ', glaccounts[int(testTargets[i])])
        print('prediction probability = ', (vote_prob2))
        
        glLabels.append(testTargets[i])
        print('K nearest neighbors: ')
        for x in range(len(neighbors)):
            print(glaccounts[int(neighbors[x][2])])
        temp_harmonicWeight = weightedVote(neighbors,results=True)[1]
        temp_dist = weightedDistance(neighbors,results=True)[1]
        for i in range(len(temp_dist)):
            print("New prediction with weight function : ", glaccounts[int(temp_dist[i][0])], " probability : ",round(temp_dist[i][1],1))
        
  
	
main()
