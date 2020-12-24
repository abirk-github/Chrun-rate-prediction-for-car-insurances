
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin #gives fit_transform method for free
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import pickle


df = pd.read_csv('cleaned_dataset.csv', low_memory=False)

# Divide the data into "features" and “labels”. 
# X variable contains all the features and y variable contains labels.
X = df.drop('kjop', axis=1)
y = df['kjop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 50)


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

classifier = LogisticRegression()

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', classifier)])

pipe.fit(X_train, y_train) 

# Prediction on test data
y_pred = pipe.predict(X_test)


confusion = confusion_matrix(y_test, y_pred)
print('Cindusion matrix: ', confusion)

print(classification_report(y_test, y_pred, target_names=['0', '1']))

# calculate the fpr and tpr for all thresholds of the classification
probs = pipe.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# save the model to disk
filename = 'model.sav'
pickle.dump(pipe, open(filename, 'wb'))
 
plt.show()