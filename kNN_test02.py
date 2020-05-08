# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:59:24 2020

@author: 18333
"""

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator

'''
Parameters:
    filename
Returns:
    returnMat - feature matrix
'''

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numOfLines = len(arrayOLines)
    returnMat = np.zeros((numOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        else:
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

'''
Parameters:
    dataSet - feature matrix
Returns:
    normDataSet - normalised feature matrix
    ranges - the range of data
    minvals - minimal of the data set
'''

def autoNorm(dataSet):
    #get min value of the column
    minvals = dataSet.min(0)
    maxvals = dataSet.max(0)
    ranges = maxvals - minvals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minvals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minvals

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

'''
Parameters:
    None
Returns:
    normDataSet - normalised feature matrix
    ranges - the range of data
    minvals - minimal of the dataset
'''

def datingClassTest():
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    #get 10% data
    hoRatio = 0.10
    normMat, ranges, minvals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    
    for i in range(numTestVecs):
        #0~numTestVecs: training set, numTestVecs~m: test set
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 4)
        print(f"Classify result: {classifierResult}\tTrue classification: {datingLabels[i]}")
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print(f"Error rate: {errorCount/ float(numTestVecs)*100}")

'''
Parameters:
    None
Returns:
    None
'''

def classifyPerson():
    resultList = ['dislike', 'small doses', 'large doses']
    precentTats = float(input("The percentage of playing video games: "))
    ffMiles = float(input("Annual frequent flight miles: "))
    iceCream = float(input("Weekly expenditure liters of ice cream: "))
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minvals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, precentTats, iceCream])
    norminArr = (inArr - minvals) / ranges
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    print(f"Maybe you {resultList[classifierResult - 1]} this person.")

'''
Parameters:
    None
Returns:
    None
'''

def showdatas(datingDataMat, datingLabels):
    font_dict = {'fontsize': 14, 'fontweight': 8.2}
    fig, axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = False, figsize = (13, 8))
    
    numOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        elif i == 2:
            LabelsColors.append('orange')
        else:
            LabelsColors.append('red')
    #flight distances, gaming spending time, s: the font size, alpha: the degree of transperant
    axs[0][0].scatter(x = datingDataMat[:,0], y = datingDataMat[:,1], color = LabelsColors, s = 15, alpha =.5)
    #set_title: set title for axes
    label0 = 'The percentage of annual frequent flight miles and the time spent on video games'
    loc = 'center'
    axs0_title_text = axs[0][0].set_title(label = label0, fontdict = font_dict, loc = loc)
    axs0_xlabel_text = axs[0][0].set_xlabel('The annual frequent flight miles', fontdict = font_dict)
    axs0_ylabel_text = axs[0][0].set_ylabel('The time spent on video games', fontdict = font_dict)
    plt.setp(axs0_title_text, size = 9, weight = 'bold', color = 'red')
    plt.setp(axs0_xlabel_text, size = 7, weight = 'bold', color = 'black')
    plt.setp(axs0_ylabel_text, size = 7, weight = 'bold', color = 'black')
    
    axs[0][1].scatter(x = datingDataMat[:,0], y = datingDataMat[:,2], color = LabelsColors, s = 15, alpha =.5)
    label1 = 'The percentage of annual frequent flight miles and the weekly ice cream consumption liters'
    axs1_title_text = axs[0][1].set_title(label = label1, fontdict = font_dict, loc = loc)
    axs1_xlabel_text = axs[0][1].set_xlabel('The annual frequent flight miles', fontdict = font_dict)
    axs1_ylabel_text = axs[0][1].set_ylabel('The weekly ice cream consumption liters', fontdict = font_dict)
    plt.setp(axs1_title_text, size = 9, weight = 'bold', color = 'red')
    plt.setp(axs1_xlabel_text, size = 7, weight = 'bold', color = 'black')
    plt.setp(axs1_ylabel_text, size = 7, weight = 'bold', color = 'black')
    
    axs[1][0].scatter(x = datingDataMat[:,1], y = datingDataMat[:,2], color = LabelsColors, s = 15, alpha =.5)
    label2 = 'The time spent on video games and the weekly ice cream consumption liters'
    axs2_title_text = axs[1][0].set_title(label = label2, fontdict = font_dict, loc = loc)
    axs2_xlabel_text = axs[1][0].set_xlabel('The weekly ice cream consumption liters', fontdict = font_dict)
    axs2_ylabel_text = axs[1][0].set_ylabel('The time spent on video games', fontdict = font_dict)
    plt.setp(axs2_title_text, size = 9, weight = 'bold', color = 'red')
    plt.setp(axs2_xlabel_text, size = 7, weight = 'bold', color = 'black')
    plt.setp(axs2_ylabel_text, size = 7, weight = 'bold', color = 'black')
    
    didntLike = mlines.Line2D([], [], color = 'black', marker = '.', markersize = 6, label = 'didntLike')
    smallDoses = mlines.Line2D([], [], color = 'orange', marker = '.', markersize = 6, label = 'smallDoses')
    largeDoses = mlines.Line2D([], [], color = 'red', marker = '.', markersize = 6, label = 'largeDoses')
    
    axs[0][0].legend(handles = [didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles = [didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles = [didntLike, smallDoses, largeDoses])
    
    plt.show()

if __name__ == '__main__':
    #filename = "datingTestSet.txt"
    #datingDataMat, datingLabels = file2matrix(filename)
    #showdatas(datingDataMat, datingLabels)
    #normDataSet, ranges, minvals = autoNorm(datingDataMat)
    #datingClassTest()
    classifyPerson()