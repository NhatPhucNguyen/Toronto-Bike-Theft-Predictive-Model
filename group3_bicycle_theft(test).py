# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:31:58 2024
"D:/Centennial-SofwareEngineering/Semester6/DataWarehousing/FinalProject/Bicycle_Thefts_Open_Data.csv"
@author: jemis
"""

import os
import folium
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Load data
DATA_SOURCE = 'Bicycle_Thefts_Open_Data.csv'
data_bicycle_theft = pd.read_csv(os.path.join(os.path.dirname(__file__),DATA_SOURCE))

# Convert coordinates
data_bicycle_theft['LATITUDE'] = data_bicycle_theft['LAT_WGS84']
data_bicycle_theft['LONGITUDE'] = data_bicycle_theft['LONG_WGS84']
#data_bicycle_theft.drop(columns=['LAT_WGS84', 'LONG_WGS84'], inplace=True)
#data_bicycle_theft.drop(columns=['X', 'Y'], inplace=True)

# Select relevant columns
relevant_columns = [
    'PRIMARY_OFFENCE', 'OCC_DATE', 'OCC_YEAR', 'OCC_MONTH', 'OCC_DOW', 'OCC_DAY',
    'OCC_DOY', 'OCC_HOUR', 'REPORT_DATE', 'REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DOW',
    'REPORT_DAY', 'REPORT_DOY', 'REPORT_HOUR', 'DIVISION', 'LOCATION_TYPE',
    'PREMISES_TYPE', 'BIKE_MAKE', 'BIKE_MODEL', 'BIKE_TYPE', 'BIKE_SPEED', 'BIKE_COLOUR',
    'BIKE_COST', 'STATUS', 'LATITUDE', 'LONGITUDE'
]

data_bicycle_theft = data_bicycle_theft[relevant_columns]

# Check for duplicates
duplicate_rows = data_bicycle_theft.duplicated().sum()
print("Duplicate Rows:", duplicate_rows)
data_bicycle_theft.drop_duplicates(inplace=True)

# Parse date columns to datetime
data_bicycle_theft['OCC_DATE'] = pd.to_datetime(data_bicycle_theft['OCC_DATE'])
data_bicycle_theft['REPORT_DATE'] = pd.to_datetime(data_bicycle_theft['REPORT_DATE'])

#Fill in the missing values of bike model with unknown
data_bicycle_theft['BIKE_MODEL'].fillna("UNKNOWN", inplace = True)
data_bicycle_theft['BIKE_MODEL'].head(30) 

#Fill in the missing values of bike color with UNKNOWN
data_bicycle_theft['BIKE_COLOUR'].fillna("UNKNOWN", inplace = True)

#Statistics
## get the bike cost mean
print(data_bicycle_theft["BIKE_COST"].mean())
## get the bike cost sum
print(np.sum(data_bicycle_theft["BIKE_COST"]))
numerical_columns = ['OCC_HOUR', 'BIKE_COST']
correlation_matrix = data_bicycle_theft[numerical_columns].corr()

print("Descriptive Statistics:")
print(data_bicycle_theft[numerical_columns].describe())
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Graphs and visualizations #

# Convert OCC_YEAR column to datetime
data_bicycle_theft['OCC_YEAR'] = pd.to_datetime(data_bicycle_theft['OCC_YEAR'], format='%Y')
bike_thefts_per_year = data_bicycle_theft.groupby(data_bicycle_theft['OCC_YEAR'].dt.year).size()
# Compute descriptive statistics
theft_stats = bike_thefts_per_year.describe()

# Print descriptive statistics
print("Descriptive Statistics of Bike Thefts per Year:\n", theft_stats)

# Visualize the number of bike thefts per year
plt.figure(figsize=(10, 6))
bike_thefts_per_year.plot(kind='bar', color='skyblue')
plt.title('Number of Bike Thefts per Year')
plt.xlabel('Year')
plt.ylabel('Number of Thefts')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Histogram
plt.figure(figsize=(12, 6))
sns.histplot(data=data_bicycle_theft, x='OCC_HOUR', bins=24, kde=True)
plt.title('Distribution of Theft Occurrences by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#Box plot highlighting the distribution of Bike Cost by Status
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_bicycle_theft, x='STATUS', y='BIKE_COST')
plt.title('Distribution of Bike Cost by Status')
plt.xlabel('Status')
plt.ylabel('Bike Cost')
plt.grid(True)
plt.show()

#Violin plot
# Set the style of seaborn
sns.set_style("whitegrid")

plt.figure(figsize=(12, 8)) # Set the figure size
sns.violinplot(data=data_bicycle_theft, x='PRIMARY_OFFENCE', y='OCC_HOUR', palette='Set2')

# Add title and labels
plt.title('Distribution of Theft Occurrences by Primary Offence', fontsize=16)
plt.xlabel('Primary Offence', fontsize=14)
plt.ylabel('Hour of the Day', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

#Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix')
plt.show()



#Geographical distribution using scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=data_bicycle_theft, x='LONGITUDE', y='LATITUDE', hue='STATUS', alpha=0.5)
plt.title('Geographical Distribution of Bicycle Thefts')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Status')
plt.grid(True)
plt.show()

#Geographical distribution using interactive map
m = folium.Map(location=[data_bicycle_theft['LATITUDE'].mean(), data_bicycle_theft['LONGITUDE'].mean()], zoom_start=12)

for _, row in data_bicycle_theft.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5,
        fill=True,
        color='red' if row['STATUS'] == 'STOLEN' else 'green',
        fill_opacity=0.7
    ).add_to(m)

m.save('bicycle_thefts_map.html')

### Data modeling
##drop null and datetime column
data = data_bicycle_theft.dropna()
data.isnull().sum()
del data['OCC_DATE']
del data['REPORT_DATE']
data.shape
data.head()
data.dtypes
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
#convert year columns
data['OCC_YEAR'] = data['OCC_YEAR'].dt.year
#label categorical features
categorical_features = ['PRIMARY_OFFENCE','OCC_MONTH','OCC_DOW','REPORT_MONTH','REPORT_DOW','DIVISION','LOCATION_TYPE','PREMISES_TYPE','BIKE_MAKE','BIKE_MODEL','BIKE_TYPE','BIKE_COLOUR','OCC_YEAR']
data['STATUS'].replace('STOLEN', 0, inplace=True)
data['STATUS'].replace(['UNKNOWN', 'RECOVERED'], 1, inplace=True)
for feature in categorical_features:
     data[feature] = encoder.fit_transform(data[feature])
x=data.iloc[:,:-1].drop('STATUS',axis=1)
y=data['STATUS']

#normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

data = scaler.fit_transform(data)
#training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

###ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    thresholds = thresholds[thresholds < 1]
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

###Logistic regresion
from sklearn.linear_model import LogisticRegression
log_classifier=LogisticRegression(random_state = 0)
log_classifier.fit(X_train,y_train)
#test
y_pred = log_classifier.predict(X_test)
from sklearn.metrics import accuracy_score
log_score = accuracy_score(y_test, y_pred)
print('Accuracy of Logistic is:', log_score)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
log_cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=log_cm,display_labels=log_classifier.classes_)
disp.plot()
plt.show()
log_prob = log_classifier.predict_proba(X_test)
plot_roc_curve(y_test, log_prob[:,1])
###Decision trees
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)
dt_cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=dt_cm,display_labels=dt_classifier.classes_)
disp.plot()
plt.show()
dt_score = accuracy_score(y_test, y_pred)
print('Decision trees accuracy score : {0:0.4f}'. format(dt_score))
##plot tree
plt.figure(figsize=(30,20))
from sklearn import tree
tree.plot_tree(dt_classifier.fit(X_test, y_test))
plt.show()
#dt_prob = dt_classifier.predict_proba(X_test)
#plot_roc_curve(y_test, dt_prob[:,1])
###Random forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
rf_cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm,display_labels=rf_classifier.classes_)
disp.plot()
plt.show()
from sklearn.metrics import accuracy_score
rf_score = accuracy_score(y_test, y_pred)
print('Random forest accuracy score : {0:0.4f}'. format(rf_score))
rf_prob = rf_classifier.predict_proba(X_test)
plot_roc_curve(y_test, rf_prob[:,1])
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
###Save model
import joblib
joblib.dump(log_classifier,'log_classifier.pkl')
joblib.dump(dt_classifier,'dt_classifier.pkl')
joblib.dump(rf_classifier,'rf_classifier.pkl')
model_columns = list(x.columns)