#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
#from voronoi import plot_voronoi

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'
class_colours = [CLASS_1_C, CLASS_2_C, CLASS_3_C]

#doug smells


MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']

def calculate_accuracy(gt_labels,pred_labels):
    counter=0
    length=len(pred_labels) #same as RES
    for i in range(length):
        if gt_labels[i]!=pred_labels[i]:
            counter+=1
            #finds the number of predicted classes that are wrong
        Acc=1-(counter/length) #how accurate classifer is as a %/100
    print(Acc)
    return Acc

def entropy(partition1i):
    p1ilength=np.size(partition1i)
    unique,counts = np.unique(partition1i, return_counts=True)
    assignment=unique[np.where(counts == max(counts))][0]
    i=0
    p1ientr=0
    while i < np.size(counts):
        p1ientr+=((-(counts[i]/p1ilength)*np.log(counts[i]/p1ilength)))
        i+=1
    p1ientr=1-p1ientr
    return unique,counts,assignment,p1ientr

def plotSelFeatures(feature1, feature2, labels):
    Class_Colours= [CLASS_1_C, CLASS_2_C, CLASS_3_C]
    Colours = []

    for i in labels:
        Colours.append(Class_Colours[int(i)-1])

    plt.scatter(feature1,feature2, c = Colours)

    plt.show()

    return

def conMat(Real, predicted):

    return 0

def feature_selection(train_set, train_labels, **kwargs):

    n_features = train_set.shape[1]
    fig, ax = plt.subplots(n_features, n_features)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

    #ax2 = plt.subplots(1,1)

    Class_Colours= [CLASS_1_C, CLASS_2_C, CLASS_3_C]
    Colours = []

    #print(train_labels)

    for i in train_labels:
        Colours.append(Class_Colours[int(i)-1])

    #print(Colours)

    #print(n_features)



    for j in range(n_features):
        for k in range(n_features):
            ax[j,k].scatter(train_set[:, j],train_set[:, k], c = Colours, s = 3)
            #ax[j,k].set_title('{} x {}'.format(j+1, k+1), fontsize = 7)
            ax[j,k].set_yticklabels([])
            ax[j,k].set_xticklabels([])

    #plt.tight_layout()
    plt.scatter(train_set[:, 9],train_set[:, 6], c = Colours)
    #plt.xaxis.label.set_size(1)
    #plt.yaxis.label.set_size(1)
    #plt.labelsize(1)
    #plotSelFeatures(train_set[:, 9], train_set[:, 6], train_labels)
    plt.show()
    fe = np.array([6, 9])



    return [fe]

def KnnNeigh(reducedTrainSet, trainLabels, reducedTestSet, k, d):
    #Function to excecute k nn algothritm
    # [IN] = reducedTrainSet, trainLabels, reducedTestSet : [NP array], k, d : [INT]
    #
    #

    predTestLabels = []
    for testData in reducedTestSet:
        neighbours= []
        neighboursClass = []

        for i, trainData in enumerate(reducedTrainSet):


            if d == 2:
                EuDistance = np.sqrt((testData[0] - trainData[0])**2 + (testData[1] - trainData[1])**2)
            elif d == 3:
                EuDistance = np.sqrt((testData[0] - trainData[0])**2 + (testData[1] - trainData[1])**2  + (testData[2] - trainData[2])**2)




            neighbours.append([EuDistance, i])


        knnNeighbours = sorted(neighbours)[:k]



        for n in knnNeighbours:
            neighboursClass.append(trainLabels[n[1]])


        (Class,frequency) = np.unique(neighboursClass , return_counts=True)

        predTestLabels.append(int(Class[np.argmax(frequency)]))






    return np.array(predTestLabels)




def knn(train_set, train_labels, test_set, k, **kwargs):
    train_set_R = train_set[:, [6,9]]
    test_set_R = test_set[:, [6,9]]

    predictions = KnnNeigh(train_set_R, train_labels, test_set_R, k, 2)
    print(Acc(test_labels, predictions))


    return predictions



def alternative_classifier(train_set, train_labels, test_set, **kwargs):

    train_set_2=train_set[:,[6,9]]
    test_set_2=test_set[:,[6,9]]
    train_labels1=train_labels
    train_labels2=train_labels
    min1=min(train_set_2[:,0])
    max1=max(train_set_2[:,0])
    min2=min(train_set_2[:,1])
    max2=max(train_set_2[:,1])
    range1=(max1-min1)
    range2=(max1-min2)
    partition1=range1/3
    partition2=range2/3
    partition1a=min1+partition1
    partition1b=min1+(2*partition1)
    partition2a=min2+partition2
    partition2b=min2+(2*partition2)
    partition1i=[]
    partition1ii=[]
    partition1iii=[]
    partition2i=[]
    partition2ii=[]
    partition2iii=[]
    f1=train_set_2[:,0]
    f2=train_set_2[:,1]
    f11=test_set_2[:,0]
    f22=test_set_2[:,1]
    i=0
    while i<125:
        if f1[i]<=partition1a:
            partition1i.append(train_labels1[i])
        if partition1a<f1[i]<=partition1b:
            partition1ii.append(train_labels1[i])
        if f1[i]>partition1b:
            partition1iii.append(train_labels1[i])
        i+=1
    j=0
    while j<125:
        if f2[j]<=partition2a:
            partition2i.append(train_labels2[j])
        if partition2a<f2[j]<=partition2b:
            partition2ii.append(train_labels2[j])
        if f2[j]>partition2b:
            partition2iii.append(train_labels2[j])
        j+=1
    childlist=[partition1i,partition1ii,partition1iii,partition2i,partition2ii,partition2iii]
    entr1i=entropy(partition1i)[3]
    ass1i=entropy(partition1i)[2]
    entr1ii=entropy(partition1ii)[3]
    ass1ii=entropy(partition1ii)[2]
    entr1iii=entropy(partition1iii)[3]
    ass1iii=entropy(partition1iii)[2]
    entr2i=entropy(partition2i)[3]
    ass2i=entropy(partition2i)[2]
    entr2ii=entropy(partition2ii)[3]
    ass2ii=entropy(partition2ii)[2]
    entr2iii=entropy(partition2iii)[3]
    ass2iii=entropy(partition2iii)[2]
    entropylist=[entr1i,entr1ii,entr1iii,entr2i,entr2ii,entr2iii]
    assignmentlist=[ass1i,ass1ii,ass1iii,ass2i,ass2ii,ass2iii]
    k=0
    alt_predictions=[]
    while k<53:
        if partition2a<f22[k]<=partition2b:
            alt_predictions.append(2)
        elif f22[k]<=partition2a:
            alt_predictions.append(2)
        elif f11[k]>partition1b:
            alt_predictions.append(1)
        elif f11[k]<=partition1a:
            alt_predictions.append(3)
        elif partition1a<f11[k]<=partition1b:
            alt_predictions.append(2)
        elif f22[k]>partition2b:
            alt_predictions.append(1)
        k+=1
    alt_predictions_array=np.array(alt_predictions)
    """calculate_accuracy(test_labels,alt_predictions_array)"""
    # write your code here and make sure you return the predictions at the end of
    # the function
    return alt_predictions_array


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    train_set_R = train_set[:, [6,9, 3]]
    test_set_R = test_set[:, [6,9, 3]]

    #3

    predictions = KnnNeigh(train_set_R, train_labels, test_set_R, k, 3)
    # write your code here and make sure you return the predictions at the end of
    # the function
    print(Acc(test_labels, predictions))
    return predictions


def Pca(trainSet, testSet):

    covMat = np.cov(trainSet, rowvar = False)
    ei = np.linalg.eig(covMat)
    eiValD = -np.sort(-ei[0])


    eiVecOrdered= np.zeros(ei[1].shape)


    vecOrder= []

    j = 0
    for i in range(np.size(ei[0])):
        if ei[0][j] == eiValD[i]:
            vecOrder.append(i)
            j+=1

    for z in range(np.size(ei[0])):
        #print(z)

        eiVecOrdered[z] = ei[1][vecOrder[z]]

    #print(eiVecOrdered[0])
    #print(ei[1][0])
    #print(ei[1][vecOrder[0]])

    w = np.array([eiVecOrdered[0], eiVecOrdered[1]])

    reducedDataTrain = np.dot(trainSet, w.transpose())
    reducedDataTest = np.dot(testSet, w.transpose())

    return reducedDataTrain,reducedDataTest

def Acc(real, precited):
    correct = 0
    for i, data in enumerate(real):
        if data == precited[i]:
            correct += 1
    return (correct/np.size(real))*100
def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):

    reducedTrainSet , reducedTestSet = Pca(train_set, test_set)
    #print(reducedTestSet)
    #print(reducedTrainSet)

    predicted = KnnNeigh(reducedTrainSet, train_labels, reducedTestSet, k, n_components)



    print(Acc(test_labels, predicted))
    #plt.scatter(reducedTrainSet[:,0], -reducedTrainSet[:,1])
    #plt.show()


    return predicted


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))


"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""


"""
>>> Combine all code

>>> mkae code pretty

>>> Fix the path stuff


>>> change argument thing

"""
