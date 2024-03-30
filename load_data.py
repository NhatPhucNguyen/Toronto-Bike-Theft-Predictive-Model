# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:07:18 2024

@author: Nhat

Police Dep Data : "D:\Centennial-SofwareEngineering\Semester6\DataWarehousing\FinalProject\Bicycle_Thefts_Open_Data.csv"
"""
### 1. Load and describe 
import pandas as pd
data_pdep = pd.read_csv('D:\Centennial-SofwareEngineering\Semester6\DataWarehousing\FinalProject\Bicycle_Thefts_Open_Data.csv')
data_pdep.columns.values
print(data_pdep.columns.values)
data_pdep.dtypes
for col in data_pdep.columns:
    print(col)   
######### get first five records
data_pdep.head(5)
######### get the shape of data
data_pdep.shape
######## get the column values
data_pdep.columns.values
###### create summaries of data
data_pdep.describe()
##### get the types of columns
data_pdep.dtypes

### 2. Statistical assessments
import numpy as np
## get the bike cost mean
print(data_pdep["BIKE_COST"].mean())
## get the bike cost sum
print(np.sum(data_pdep["BIKE_COST"]))
## correlations
np.corrcoef(data_pdep["X"],data_pdep["Y"])


### 3. Missing data evaluation
# Fill the missing values with zeros
data_pdep.fillna(0,inplace=True)
data_pdep.head()
# Fill the missing values with "missing"
data_pdep.fillna("missing",inplace=True)
data_pdep.head(30)
# use the average to fill in the missing OCC_HOUR
data_pdep['OCC_HOUR'].head(30)
## get the OCC_HOUR mean
print(data_pdep['OCC_HOUR'].mean())
##
data_pdep['OCC_HOUR'].fillna(data_pdep['OCC_HOUR'].mean(),inplace=True)
data_pdep['OCC_HOUR'].head(30)

### 4. Graph and visualization
import matplotlib.pyplot as plt
#create a scatterplot
fig_pdep = data_pdep.plot(kind='scatter',x='BIKE_COST',y='BIKE_SPEED')
# Save the scatter plot
fig_pdep.figure.savefig('D:\Centennial-SofwareEngineering\Semester6\DataWarehousing\FinalProject\PD_ScatterPlot.pdf')
#plot a histrogram
plt.hist(data_pdep['OCC_HOUR'],bins=24)
plt.xlabel('HOUR')
plt.ylabel('Case')
plt.title('Frequency of bike theft')
plt.savefig('D:\Centennial-SofwareEngineering\Semester6\DataWarehousing\FinalProject\PD_HistoGraph.pdf')
plt.show()
#Plot a Box Plot
plt.boxplot(data_pdep['OCC_HOUR'])
plt.ylabel('Occur Hour')
plt.title('Box Plot of cases in hour')
plt.show()
#Piechart
data_pdep["PREMISES_TYPE"].value_counts(dropna=False).plot(kind="pie",figsize=(8,8))

import numpy as np
def corrcoeff(df,var1,var2):
    df['corrn']=(df[var1]-np.mean(df[var1]))*(df[var2]-np.mean(df[var2]))
    df['corrd1']=(df[var1]-np.mean(df[var1]))**2
    df['corrd2']=(df[var2]-np.mean(df[var2]))**2
    corrcoeffn=df.sum()['corrn']
    corrcoeffd1=df.sum()['corrd1']
    corrcoeffd2=df.sum()['corrd2']
    corrcoeffd=np.sqrt(corrcoeffd1*corrcoeffd2)
    corrcoeff=corrcoeffn/corrcoeffd
    return corrcoeff
print(corrcoeff(data_pdep,'X','Y'))

import matplotlib.pyplot as plt
plt.plot(data_pdep['BIKE_SPEED'],data_pdep['BIKE_COST'],'ro')
plt.title('BIKE_SPEED vs BIKE_COST')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
feature_cols = ['BIKE_SPEED']
X = data_pdep[feature_cols]
Y = data_pdep['BIKE_COST']
X.fillna(0,inplace=True)
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
lm = LinearRegression()
lm.fit(trainX, trainY)
print (lm.intercept_)
print (lm.coef_)
zip(feature_cols, lm.coef_)
[('BIKE_SPEED',50)]
lm.score(trainX, trainY)
lm.predict(testX)
