# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 18:18:27 2021

@author: HP

"""
"""
TASK  :  TO CHECK WHETHER SETTING UP FOODTRUCK IN JAIPUR WOULD BE PROFITABLE OR not

"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
#reading the data
foodtruck = pd.read_csv('Foodtruck.csv')

#Listing columns
foodtruck.columns.tolist()

#Checking missing values

foodtruck.isnull().sum()

features = foodtruck['Population'].values
features = features.reshape(97, 1)

labels = foodtruck['Profit'].values
labels = labels.reshape(97, 1)

regressor = LinearRegression()

predict_sales = regressor.fit(features, labels)

print(regressor.coef_)

# To predict the sales of Jaipur with current population -3.073 million

predict_sales = regressor.predict([[3.073]])

## There would be loss in setting up an outlet in Jaipur as predicted value of sales is negative.
