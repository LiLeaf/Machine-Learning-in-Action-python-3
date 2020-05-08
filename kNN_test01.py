# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:56:36 2020

@author: Leaf
"""

import numpy as np

'''
createDataSet(): create the dataset of kNN
Parameters:
    none
Returns:
    group - DataSet
    labels - classification label
'''

def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''
kNN algorithm and classifier
Parameters:
    inX - the dataset used for classify(test set)
    dataSet - the dataset used for training data(trainning set)
    labels - classification labels
    k - parameter of kNN algorithm, choose k points with minimum distances.
Returns:
    sortedClassCount[0][0] - classification result
'''

def classify0(inX, dataSet, labels, k):
    #the number of row
    dataSetSize = dataSet.shape[0]
    #repeat on row "dataSetSize" times, on column 1 time.
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key, default=None), get identified value of key, if not exit, return default value
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #key = operator.itemgetter(0), sort as key order; key = operator.itemgetter(1), sort as value order
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    #create dataset
    group, labels = createDataSet()
    #test dataset
    test = [101, 20]
    #kNN classification
    test_class = classify0(test, group, labels, 3)
    #print result
    print(test_class)