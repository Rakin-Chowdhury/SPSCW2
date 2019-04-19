#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

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


MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']

def plotSelFeatures(feature1, feature2, labels):
    Class_Colours= [CLASS_1_C, CLASS_2_C, CLASS_3_C]
    Colours = []

    for i in labels:
        Colours.append(Class_Colours[int(i)-1])

    plt.scatter(feature1,feature2, c = Colours)

    plt.show()

    return

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

def KnnNeigh(reducedTrainSet, trainLabels, reducedTestSet, k):

    predTestLabels = []
    for testData in reducedTestSet:
        neighbours= []
        neighboursClass = []

        for i, trainData in enumerate(reducedTrainSet):
            EuDistance = np.sqrt((testData[0] - trainData[0])**2 + (testData[1] - trainData[1])**2)
            neighbours.append([EuDistance, i])


        knnNeighbours = sorted(neighbours)[:k]



        for n in knnNeighbours:
            neighboursClass.append(trainLabels[n[1]])


        predTestLabels.append(int(round(np.mean(neighboursClass))))


    return np.array(predTestLabels)




def knn(train_set, train_labels, test_set, k, **kwargs):
    train_set_R = train_set[:, [6,9]]
    test_set_R = test_set[:, [6,9]]

    predictions = KnnNeigh(train_set_R, train_labels, test_set_R, k)
    print(Acc(test_labels, predictions))


    return predictions



def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


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

    predicted = KnnNeigh(reducedTrainSet, train_labels, reducedTestSet, k)

    #print(Acc(test_labels, predicted))
    plt.scatter(reducedTrainSet[:,0], -reducedTrainSet[:,1])
    plt.show()


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
