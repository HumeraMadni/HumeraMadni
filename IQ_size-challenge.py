# -*- coding: utf-8 -*-
#Task 1:
"""
Write a Python code that fulfills the following specification.
dataset: IQ_size.csv

It Contains the details of 38 students, where

Column 1: The intelligence (PIQ) of students

Column 2:  The brain size (MRI) of students (given as count/10,000).

Column 3: The height (Height) of students (inches)

Column 4: The weight (Weight) of student (pounds)

Task01:
What is the IQ of an individual with a given 
brain size of 90, height of 70 inches, and weight 150 pounds ? 

Task02:
Are a person's brain size and body size (Height and weight) predictive 
of his or her intelligence?

"""
##Import libraries

import pandas as pd
import numpy as np

## Read the dataseet

iq_data= pd.read_csv("IQ_Size.csv")

#check the shape

iq_data.shape
#Check the missing values
iq_data.isnull().any(axis = 0)

## The dataset contains no missing values

## lets split our features and labels
iq_data.head()

features = iq_data.iloc[:, 1:4].values
labels = iq_data.iloc[:, 0].values

## to predict the IQ of an individual will use LinearRegression 

## splitting the data into train and test

from sklearn.model_selection import train_test_split

train_features, test_features,train_label, test_label = train_test_split(features, labels, random_state =0, test_size =0.2)

from sklearn.linear_model import LinearRegression

#Instantiate linear regression model

regressor = LinearRegression()

regressor.fit(train_features, train_label)
#task1:
"""
What is the IQ of an individual with a given 
brain size of 90, height of 70 inches, and weight 150 pounds ?
"""

pred = regressor.predict([[90, 70, 150]])

print(round(int(pred), 2))
"""
The IQ of an individual with the above given measures would be 105
"""

'''
Task02:
Are a person's brain size and body size (Height and weight) predictive 
of his or her intelligence?'''

## For this task we need to use backward elimination method to find the most important feature

import statsmodels.api as sm

features = sm.add_constant(features)
features_sm = features[:, [0,1,2,3]]

labels = labels.reshape(len(labels), 1)


stats_method = sm.OLS(labels, features_sm)

## fit the features and label in ols model

stats_method = stats_method.fit()

print(stats_method.summary())

## The p-value for weight is seen in summary as 0.99 so we will drop weight column

## we will fit the model again by dropping the weight column

features_sm =  features[:, [0,1,2]]
stats_method = sm.OLS(labels, features_sm)
stats_method = stats_method.fit()
print(stats_method.summary())

## The p-value for const is close 0.05 so we will drop this one to be more accurate

features_sm =  features[:, [1,2]]
stats_method = sm.OLS(labels, features_sm)
stats_method = stats_method.fit()
print(stats_method.summary())

## the p-value for height is 0.076 so we will drop it


features_sm =  features[:, 1]
stats_method = sm.OLS(labels, features_sm)
stats_method = stats_method.fit()
print(stats_method.summary())

