# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:28:53 2020

@author: 18333
"""


import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

"""
convert the 32*32 binary images to 1*1024 vector
Parameters:
    filename
Returns:
    returnVect - the return vectors of 1*1024
"""

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(linestr[j])
    return returnVect

"""
test of the handwritten numbers classification
Parameters:
    None
Returns:
    None
"""

def handwrittenClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNum = int(fileNameStr.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
    neigh = kNN(n_neighbors = 3, algorithm = 'auto')
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNum = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        classifierResult = neigh.predict(vectorUnderTest)
        print(f'Classifier result is {classifierResult}. The real result is {classNum}.')
        if classifierResult != classNum:
            errorCount += 1
    print(f"The total wrong number is {errorCount}. The error rate is {errorCount/mTest * 100:.3f}%.")
    return

if __name__ == '__main__':
    handwrittenClassTest()