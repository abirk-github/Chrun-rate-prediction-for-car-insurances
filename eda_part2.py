import pandas as pd 
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from feature_engine import categorical_encoders as ce

df = pd.read_csv('case_fremtind.csv', low_memory=False)

# Some of the features are columns that are of the categorical form: 
# E.g. kunde_kategori_1 --> 1, 2, 3, 4, etc...
# Intuitionally, if you can group data together in your head fairly easily and represent it with a string, 
# there’s a chance it’s part of a category.
print('Df types', df['kunde_kategori_1'].dtypes)

# Exploring categorical data
print('kunde_kategori_1', df['kunde_kategori_1'].head())

# Exploring numerical data
print('n_pers_husstand', df['n_pers_husstand'].head())

# Exploring grouped data, numerical increasing
print('andel_inntektsgruppe_1', df['andel_inntektsgruppe_1'].head())
print('andel_yrke_middelklasse', df['andel_yrke_middelklasse'].head(10))

# Label encoding operations for grouped numerical increasing data 
data = asarray(df['andel_yrke_middelklasse'].head(10))
encoder = LabelEncoder()
result = encoder.fit_transform(data)
print('Label encoded andel_yrke_middelklasse', result)

print('antall_brann_forsikring', df['antall_brann_forsikring'].head(10))
data = asarray(df['antall_brann_forsikring'].head(10)).reshape(-1,1)
encoder = LabelEncoder()
result = encoder.fit_transform(data)
print('Label encoded antall_brann_forsikring', result)

# One-hot encoding operations for categorical kunde_kategori_1 values becasue no relationship exists between them
print('kunde_kategori_1', df['kunde_kategori_1'].head(10))
data = asarray(df['kunde_kategori_1'].head(10)).reshape(-1,1)
# sparse = True --> 1D Array form 
encoder = OneHotEncoder(sparse=True)
result = encoder.fit_transform(data)
print('One-hot encoded kunde_kategori_1', result)
# sparse = False --> 2D Array form 
encoder = OneHotEncoder(sparse=False)
result = encoder.fit_transform(data)
print('One-hot encoded kunde_kategori_1', result)
print('Type kunde_kategori_1', df['kunde_kategori_1'])

# Get a complpete overview of data types of the columns  
print('Data types: ', df.dtypes)

# Deductions:
# kunde_kategori_1: categorical data            --> One-Hot encoding
# n_hus and n_pers_husstand                     --> No encoding (numerical values)
# The rest: Grouped numerical increasing values --> Label encoding

# Label encoded
print('andel_yrke_middelklasse', df['andel_yrke_middelklasse'].head(10))
data = asarray(df['andel_yrke_middelklasse'].head(10)).reshape(-1,1)
encoder = LabelEncoder()
result = encoder.fit_transform(data)
print('Label encoded andel_yrke_middelklasse', result)

# Ordinal encoded
print('andel_yrke_middelklasse', df['andel_yrke_middelklasse'].head(10))
data = asarray(df['andel_yrke_middelklasse'].head(10)).reshape(-1,1)
encoder = OrdinalEncoder()
result = encoder.fit_transform(data)
print('Ordinal encoded andel_yrke_middelklasse', result)


# Transform the categorical variables to numerical representation
df = df.dropna()
labelencoder = LabelEncoder()
for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name] = labelencoder.fit_transform(df[col_name])


print('andel_yrke_middelklasse', df['andel_yrke_middelklasse'].head(10))

# Write changed df to a seperate csv file named task3.csv
df.to_csv('cleaned_dataset.csv')

print(df.head(10))
# Obs! Alder!!! Sjekk min, max og median. Sjekk features n;yere