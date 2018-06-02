# -*- coding: utf-8 -*-


import numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree





import cPickle as pickle
train_positives = pickle.load(open('training_set_positives.p', 'rb'))
train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))




#PART a)Feature Extraction
def feature1(x):
    """This feature computes the proportion of black squares to the
       total number of squares in the grid.
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature1_value: type-float
       """

    temp = []
    totalLength = []
    # x = train_positives.values()[0]
    # print x
    for row in x:
        for value in row:
            totalLength.append(value)
            if value == 1:
                temp.append(value)

    lengthOfValue = float(len(temp))
    number0fSquares = float(len(totalLength))
    feature1_value = lengthOfValue / number0fSquares



    return feature1_value

def feature2(x):
    """This feature computes the sum of the max of continuous black squares
       in each row
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature2_value: type-float
       """
    longestNumbersLists= []

    for row in x:
        longest = 0
        current = 0
        for value in row:

            if value == 1:
                current += 1
            else:
                longest = max(longest, current)
                current = 0
        longestNumbersLists.append(max(longest,current))
    feature2_value=float(sum(longestNumbersLists))


    return feature2_value


# PART b) Preparing Data
def part_b():

    X=[]

    for x in train_positives.values():
        features1,features2=feature1(x),feature2(x)
        X.append((features1,features2))





    for x in train_negatives.values():

        features1, features2 = feature1(x), feature2(x)
        X.append((features1, features2))


    X=np.array(X)


    lable1=np.ones(len(train_positives.values()))
    lable2=np.zeros(len(train_negatives))
    y=np.concatenate([lable1,lable2])

    return X,y
# PART c) Classification with SGDClassifier

clf = linear_model.SGDClassifier(alpha=0.001,random_state=0,max_iter=20)


def part_c(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).

    """
    features1,features2=[],[]
    features1.append(feature1(x))
    features2.append(feature2(x))
    x_meze=np.array([features1,features2]).reshape(1,-1)


    X,y=part_b()

    clf.fit(X, y)
    #

    predicted_class= clf.predict(x_meze)
    # print accuracy_score(y, predicted_class, normalize = True)*100

    #
    return predicted_class


#
#
# PART d) Assess the performance of the classifier in part c
def part_d():

    X,y=part_b()
    y_train_pred = cross_val_predict(clf, X, y,cv=3)




    precision,recall= precision_score(y, y_train_pred),recall_score(y, y_train_pred)
    confusionMatrix=confusion_matrix(y, y_train_pred)



    return [precision,recall,confusionMatrix]


# PART e) Classification with RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=0)

def part_e(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    features1, features2 = [], []
    features1.append(feature1(x))
    features2.append(feature2(x))
    x_meze = np.array([features1, features2]).reshape(1, -1)

    X, y = part_b()
    forest_clf.fit(X, y)


    predicted_class = forest_clf.predict(x_meze)



    return predicted_class


#
#
# PART f) Assess the performance of the classifier in part e
def part_f():
    # predicted_class = []
    # for i in train_positives.values():
    #     predicted_class.append(part_e(i))
    # for i in train_negatives.values():
    #     predicted_class.append(part_e(i))
    X, y = part_b()
    y_train_pred_f = cross_val_predict(forest_clf, X, y, cv=3)
    print accuracy_score(y, y_train_pred_f)

    precision, recall = precision_score(y, y_train_pred_f), recall_score(y, y_train_pred_f)
    confusionMatrix = confusion_matrix(y, y_train_pred_f)

    return [precision, recall, confusionMatrix]






# PART g) Your Own Classification Model


clf_pf = tree. DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,
                           max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
                           min_samples_split=3, min_weight_fraction_leaf=0.0)


def custom_model(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).

    """



    X,y=part_b()


    features1, features2 = [], []
    features1.append(feature1(x))
    features2.append(feature2(x))
    x_meze = np.array([features1, features2]).reshape(1, -1)






    clf_pf.fit(X, y)


    predicted_class = clf_pf.predict(x_meze)




    return predicted_class

#if you want to run this
def custom_model_assest():
    X, y = part_b()
    y_train_pred = cross_val_predict(clf_pf, X, y, cv=3)

    precision, recall = precision_score(y, y_train_pred), recall_score(y, y_train_pred)
    confusionMatrix = confusion_matrix(y, y_train_pred)

    return [precision, recall, confusionMatrix]



