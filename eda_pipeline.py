import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split
#from multilabel import multilabel_train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import TransformerMixin #gives fit_transform method for free

df = pd.read_csv('cleaned_dataset.csv', low_memory=False)

X = df.drop('kjop', axis=1)
y = df['kjop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numeric_features = df.select_dtypes(include=['int64'], exclude=['object']).drop(['kjop', 'kunde_kategori_1'], axis=1).columns
categorical_features = df.select_dtypes(include=['int64']).drop(['kjop', 'n_pers_husstand', 'n_hus'], axis=1).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

pipeline_models = [
    LogisticRegression(),
    MLPClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
    ]
for clf in pipeline_models:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', clf)])
    pipe.fit(X_train, y_train)   
    #print(regressor)
    print("model score: %.3f" % pipe.score(X_test, y_test)) 
    

