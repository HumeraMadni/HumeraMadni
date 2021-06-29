# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 22:20:24 2021

@author: HP
"""


import pandas as pd
import numpy as np
import re

df = pd.read_csv('thanksgiving.csv', encoding = "Windows 1252")

## replace the column names by number codes
columns = df.columns.tolist()



number_codes = np.arange(0,len(columns))
number_codes = [x for x in number_codes]

# Fetching the columns names for further reference
number_code_mapping = dict(zip(number_codes, columns))


#Initializing the dataframe with the codes of the column
df.columns = number_codes


# Fetching the data of the people who perform thanksgiving.
#print(number_code_mapping)

people_celebrating = df[df[1] == 'Yes']


## Total 980 out of 1058 have been celebrating thanksgiving

#check for missing values

missing_values = df.isnull().sum(axis = 0)

# filling out the missing values wiht word 'missing'

fill_missing_values = df.fillna('missing', inplace = True)

## Analyzing state/area/region and income based what is consumed in their thanksgiving


# income based
income_based = df.groupby(63)

#area_type
area_based = df.groupby(60)
area_based_thanksgiving_celebration = people_celebrating.groupby(60)


#Analysis of the sauces prefered by each incomes group people
sauce_analysis = df.groupby(8)[63].value_counts()

#What is your gender? convert column to numeric values. 
gender = df[62].value_counts()

def gender(value):
    if value == 'Female':
        value = 1
    else:
        value = 0
    return value
        
df[62] = df[62].apply(gender)

print (df[62].value_counts(dropna = False))

#income cleanup
 # First the replacement function  
df[63] = df[63].replace(['Prefer not to answer', 'missing'], ["0","0"])
print(df[63])
income_values = df[63].value_counts()

##Using regex to clean      

regex = re.compile("\d+\W*\d+")


# Will apply filter to income
def income_filter(value):
    value = regex.findall(value)
    value = [int(x.replace(",", "")) for x in value]
    return sum(value)/(len(value)+0.1)

## applying the income_filter function created to dataframe
    
df[63] = df[63].apply(income_filter)

# Fetching the average incomes for each type sauces

average_income_per_sauce = df.groupby(8)[63].mean()

        
print(average_income_per_sauce)

## Visualizing the average_income_per_sauce
average_income_per_sauce.plot.bar()