import pandas as pd 
import matplotlib.pyplot as plt
import missingno as msno

df = pd.read_csv('case_fremtind.csv', low_memory=False)

print('Columns:', df.columns)

# Create correlation matrix wrt kjop for checking compatibility with linear regression
correlation_whole_set = df.corr(method="pearson")['kjop'][:]
print("Correleation for whole set wrt to kjop", correlation_whole_set)
print("Correlation size", correlation_whole_set.shape)

# Checking for amount of NaNs (i.e. missing data)
print("Nr of NaNs: ", df.isnull().sum().sum())

# Visualizing the missing data. 
msno.matrix(df, labels=True)

# From the matrix, 
# we can easily deduct that there is no need for imputation or data cleaning 
# operations


plt.show()