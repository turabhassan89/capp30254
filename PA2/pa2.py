import pandas as pd
import urllib.request
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
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
    data = read_data()
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

def logistic_regression_model():
    '''
    builds a logistic regression model using sklearn
    '''

    data = fill_data()
    independent_vars = data.columns.difference([DEPENDENT_VARIABLE])

    x_train, x_test, y_train, y_test  =  generate_data(data, independent_vars)


    model = LogisticRegression()
    model.fit(x_train, y_train)
    predicted_y_test = model.predict(x_test)

    print ("Accuracy: ", metrics.accuracy_score(y_test, predicted_y_test))
    print ("Precision: ", metrics.precision_score(y_test, predicted_y_test))
    print ("F1 score is (1 is the best): ",metrics.f1_score(y_test, predicted_y_test))




