from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import csv


'''
The constants will be changed everytime with the new data and with the demands
and requirements of the data
'''

FILENAME = 'cs-training.csv'
DATA = 'creditdata'
DISCRETIZE_VARIABLES = ['age', 'MonthlyIncome']
NO_OF_BINS = 5
BIN_LABELS = list ( range ( 1,NO_OF_BINS+1 ) )
DEPENDENT_VARIABLE = 'SeriousDlqin2yrs'
TEST_SIZE = .2

print('turab')

def define_clfs_params():

    clfs = {
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linearsvc', probability=True, random_state=0),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3), 
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
            }

    grid = { 
        'RF': {'n_estimators': [1,10,100], 'max_depth': [1,5,10], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'SVM':{'C' :[1,10],'kernel':['linearsvc']},
        'KNN':{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform'],'algorithm': [ 'auto'] },
        'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
               }

    return clfs,grid       

def read_data():
    '''
    Takes in the Filename of the csv file we want to read and returns a 
    pandas dataframe.
    '''

    rv = pd.DataFrame.from_csv(FILENAME)

    return rv

def explore_data():
    '''
    General Function to explore the data and generate summary stats
    '''
    data = fill_data()
    for column in data.columns.values.tolist():
        data.hist(column)
        plt.savefig(column)
    summarystats = data.describe ( percentiles = [] ).transpose()
    summarystats.to_csv('summarystats')
    mode = data.mode(numeric_only = True )
    mode.to_csv('mode')
    missingvalues = data.isnull().sum()
    missingvalues.to_csv('missingvalues')

def fill_data():
    '''
    General Function to fill in the missing values, currently outfitted to
    just fill the empty values with the mean of the column.
    After filling in the missing values, returns the data. 
    '''

    data = read_data()
    rv = data.fillna ( data.mean() )

    return rv

def create_bins():
    '''
    Makes bins for continous variables and puts each row in the bin in which
    its coming. After putting the data in the bins, returns the data
    '''

    data = fill_data()
    for variable in DISCRETIZE_VARIABLES:
        bins = pd.DataFrame( pd.cut(data[variable], bins = NO_OF_BINS,  labels = BIN_LABELS ) )
        newcolname = 'bin' + variable
        bins = bins.rename(columns = { variable: newcolname } )
        data = pd.concat( [data, bins], axis=1, join_axes=[data.index] )
        dummies = pd.DataFrame(create_dummies(data, newcolname, variable))
        data = pd.concat( [data, dummies], axis=1, join_axes=[data.index] )
            
    #print(dummies)
    return data

def create_dummies(dataframe, colname, collabel):
    '''
    Takes in a pandas dataframe generates dummies of the col specified and appends
    the data frame with the dummies and returns the data frame
    '''

    dummies =  pd.get_dummies(dataframe[colname], prefix = collabel, prefix_sep = '_' )

    return dummies

def generate_data(dataframe, independent_vars):
    '''
    Makes a testing and traning dataset from the dataframe based on the independent
    variables, dependent_variables and test size.
    '''
    x_train, x_test, y_train, y_test = \
    train_test_split(dataframe[independent_vars], dataframe[DEPENDENT_VARIABLE],\
    test_size = TEST_SIZE, random_state = 0)

    return x_train, x_test, y_train, y_test

def clf_loop():
    data = fill_data()
    clfs,grid = define_clfs_params()
    models_to_run=['KNN','RF','LR','DT','SVM','AB']
    independent_vars = data.columns.difference([DEPENDENT_VARIABLE])
    count = 1

    with open ('output/resultstable.csv', 'w') as csvfile:
        results = csv.writer(csvfile, delimiter=',')
        results.writerow(['Model', 'Parameters', 'area_under_curve','accuracy','precision','recall','f1'])
        for n in range(1, 2):
            X_train, X_test, y_train, y_test = train_test_split ( data[independent_vars], data[DEPENDENT_VARIABLE],\
                test_size = TEST_SIZE, random_state = 0 )
            for index,clf in enumerate([clfs[x] for x in models_to_run]):
                print ( models_to_run[index] )
                parameter_values = grid[models_to_run[index]]
                for p in ParameterGrid(parameter_values):
                    try:
                        clf.set_params(**p)
                        print (clf)
                        clf.fit(X_train, y_train)
                        y_pred_probs = clf.predict(X_test)
                        #threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
                        #print threshold
                        auc,accuracy,precision,recall,f1 = evaluate(y_test,y_pred_probs)
                        plot_precision_recall_n(y_test,y_pred_probs,models_to_run[index]+str(count) )
                        table_row = [models_to_run[index],clf,auc,accuracy,precision,recall,f1]
                        results.writerow(table_row)
                    except IndexError as e:
                        print ( 'Error:',e ) 
                        continue
                    count += 1    

'''
def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)
'''


def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    plt.figure()
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    

    plt.title(model_name)
    plt.savefig( 'output/' + model_name )
    plt.close()  

def evaluate(y_validate, pred_probs):
    
    area_under_curve = metrics.roc_auc_score(y_validate, pred_probs)
    accuracy = metrics.accuracy_score(y_validate, pred_probs)
    precision = metrics.precision_score(y_validate, pred_probs)
    recall = metrics.recall_score(y_validate, pred_probs)
    f1 = metrics.f1_score(y_validate, pred_probs)

    return area_under_curve,accuracy,precision,recall,f1

clf_loop()

'''
def logistic_regression_model():
    
    builds a logistic regression model using sklearn


    data = fill_data()
    independent_vars = data.columns.difference([DEPENDENT_VARIABLE])

    x_train, x_test, y_train, y_test  =  generate_data(data, independent_vars)


    model = LogisticRegression()
    model.fit(x_train, y_train)
    predicted_y_test = model.predict(x_test)

    print ("Accuracy: ", metrics.accuracy_score(y_test, predicted_y_test))
    print ("Precision: ", metrics.precision_score(y_test, predicted_y_test))
    print ("F1 score is (1 is the best): ",metrics.f1_score(y_test, predicted_y_test))
'''



