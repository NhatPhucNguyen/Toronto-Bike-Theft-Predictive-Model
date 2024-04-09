
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:14:06 2024

@author: Harpreet
"""
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
# Set the color palette
sns.set_palette(['#4B8BBE', '#CCCCCC'])  # Blue and grey color scheme

# Line Plot: Trends over time (for example, thefts per year)
plt.figure(figsize=(12, 6))
sns.lineplot(data=bike_thefts_per_year)
plt.title('Trend of Bike Thefts per Year')
plt.xlabel('Year')
plt.ylabel('Number of Thefts')
plt.grid(True)
plt.show()

# Pie Chart: Proportions of thefts by premises type
plt.figure(figsize=(8, 8))
data_bicycle_theft['PREMISES_TYPE'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#4B8BBE', '#CCCCCC'])
plt.title('Proportion of Bike Thefts by Premises Type')
plt.ylabel('')  # Hide the y-label
plt.show()

# Count Plot: Thefts by day of the week
plt.figure(figsize=(12, 6))
sns.countplot(x='OCC_DOW', data=data_bicycle_theft, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Bike Thefts by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Thefts')
plt.xticks(rotation=45)
plt.show()

# Enhanced Scatter Plot: Cost vs. Speed with regression line
plt.figure(figsize=(10, 8))
sns.regplot(x='BIKE_COST', y='BIKE_SPEED', data=data_bicycle_theft, scatter_kws={'alpha':0.5}, line_kws={'color': 'grey'})
plt.title('Bike Cost vs. Bike Speed with Regression Line')
plt.xlabel('Bike Cost')
plt.ylabel('Bike Speed')
plt.grid(True)
plt.show()

# Updating the existing visuals with the chosen color palette
# Histogram updated with blue color
plt.figure(figsize=(12, 6))
sns.histplot(data=data_bicycle_theft, x='OCC_HOUR', bins=24, color='#4B8BBE', kde=True)
plt.title('Distribution of Theft Occurrences by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Box plot updated with blue and grey
plt.figure(figsize=(10, 6))
sns.boxplot(x='STATUS', y='BIKE_COST', data=data_bicycle_theft, palette=['#4B8BBE', '#CCCCCC'])
plt.title('Distribution of Bike Cost by Status')
plt.xlabel('Status')
plt.ylabel('Bike Cost')
plt.grid(True)
plt.show()

### Data modeling
##drop null and datetime column
data = data_bicycle_theft.dropna()
data.isnull().sum()
del data['OCC_DATE']
del data['REPORT_DATE']
data.shape
data.head()
data.dtypes
result = data.loc[(data["STATUS"]=="RECOVERED")]
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
#convert year columns
data['OCC_YEAR'] = data['OCC_YEAR'].dt.year
data["PRIMARY_OFFENCE"].value_counts()
#remove unknown status
data = data[data.STATUS != "UNKNOWN"]
data['STATUS'].value_counts() #30187 STOLEN
#resolve imbalance
from sklearn.utils import resample

# Separate majority and minority classes
data_majority = data[data.STATUS=='STOLEN']
data_minority = data[data.STATUS=='RECOVERED']
 
# Upsample minority class
data_minority_upsampled = resample(data_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=31087,    # to match majority class
                                 random_state=123)
data_upsampled = pd.concat([data_majority, data_minority_upsampled])
#label main feature
data_upsampled['STATUS'].replace('STOLEN', 0, inplace=True)
data_upsampled['STATUS'].replace('RECOVERED', 1, inplace=True)
#label categorical features
categorical_features = []
num_attrs = []
for col, col_type in data.dtypes.items():
    if col != 'STATUS':
        if col_type == 'object':
          categorical_features.append(col)
        else:
          num_attrs.append(col)
'''for feature in categorical_features:
     data[feature] = encoder.fit_transform(data[feature])'''
data_no_feature = data_upsampled.drop("STATUS",axis=1)
'''data_dum = pd.get_dummies(data_no_feature,columns=categorical_features,dummy_na=False)
column_names = data_dum.columns
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled_data = sc.fit_transform(data_dum)'''
# build a pipeline for preprocessing the numerical attributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# numeric_df = df_upsampled['Cost_of_Bike']
num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
pipeline_transform = ColumnTransformer([
        ("num", num_pipeline, num_attrs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])
x=pipeline_transform.fit_transform(data_no_feature)
y=data_upsampled['STATUS']

#normalize data
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()

#scaled_data = scaler.fit_transform(scaled_data)
#training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
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
from sklearn.metrics import accuracy_score,classification_report
log_score = accuracy_score(y_test, y_pred)
print('Accuracy of Logistic is:', log_score)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
log_cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=log_cm,display_labels=log_classifier.classes_)
disp.plot()
plt.show()
log_prob = log_classifier.predict_proba(X_test)
plot_roc_curve(y_test, log_prob[:,1])
classification_report(y_test, y_pred)
###Decision trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
dt_cm = confusion_matrix(y_test,dt_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=dt_cm,display_labels=dt_classifier.classes_)
disp.plot()
plt.show()
dt_score = accuracy_score(y_test, dt_pred)
classification_report(y_test,dt_pred)
dt_pred_prob = dt_classifier.predict_proba(X_test)
plot_roc_curve(y_test, dt_pred_prob[:,1])
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
from sklearn.metrics import classification_report
rf_classifier = RandomForestClassifier(random_state=0,max_depth=5)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
rf_cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm,display_labels=rf_classifier.classes_)
disp.plot()
plt.show()
from sklearn.metrics import accuracy_score,classification_report
rf_score = accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
print('Random forest accuracy score : {0:0.4f}'. format(rf_score))
rf_prob = rf_classifier.predict_proba(X_test)
plot_roc_curve(y_test, rf_prob[:,1])
classification_report(y_test, y_pred)
###Save model
import joblib
joblib.dump(log_classifier,'log_classifier.pkl')
joblib.dump(dt_classifier,'dt_classifier.pkl')
joblib.dump(rf_classifier,'rf_classifier.pkl')
joblib.dump(pipeline_transform,'pipeline_transform.pkl')