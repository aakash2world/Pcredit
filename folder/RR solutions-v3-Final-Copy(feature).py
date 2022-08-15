#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Setting my directory path
import os
os.chdir('F:\RR Solutions')


# In[4]:


# Import required python packages for the analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


# Read the data file - Application_Data.csv is a cleaned data file sourced from the below URl
# https://www.kaggle.com/datasets/caesarmario/application-data?resource=download
df = pd.read_csv("Application_Data.csv")


# In[6]:


# Checking the data file details such as datatypes, count
df.info()


# In[7]:


# checking if the data file contains null or empty records - as this is cleaned data file version so null records are found
df.isnull().sum()


# In[8]:


#we are filtering the columns that have non numeric values to see if they are useful
ot = pd.DataFrame(df.dtypes =='object').reset_index()
object_type = ot[ot[0] == True]['index']
object_type


# In[9]:


num_type = pd.DataFrame(df.dtypes != 'object').reset_index().rename(columns =  {0:'yes/no'})
num_type = num_type[num_type['yes/no'] ==True]['index']


# In[10]:


a = df[object_type]['Applicant_Gender'].value_counts()
b = df[object_type]['Income_Type'].value_counts()
c = df[object_type]['Education_Type'].value_counts()
d = df[object_type]['Family_Status'].value_counts()
e = df[object_type]['Housing_Type'].value_counts()
e = df[object_type]['Job_Title'].value_counts()

print( a,"\n",b,'\n', c, '\n', d, '\n', e)


# In[11]:


# transforming 'Object' data types to integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for x in df:
    if df[x].dtypes=='object':
        df[x] = le.fit_transform(df[x])


# In[65]:


# checking the data details after converting object data types to integers
df.info()


# In[66]:


df.head(10)


# In[67]:


df[num_type].head()


# In[68]:


# describe complete details of the data set 
df.describe()


# In[69]:


# checking correlations between the varaibales
df.corr()


# In[81]:


import seaborn as sns
fig, ax= plt.subplots(nrows= 3, ncols = 3, figsize= (14,6))
sns.scatterplot(x='Status', y='Applicant_Gender', data=df, ax=ax[0][0], color= 'orange')
sns.scatterplot(x='Status', y='Total_Income', data=df, ax=ax[0][1],color= 'orange')
sns.scatterplot(x='Status', y='Applicant_Age', data=df, ax=ax[0][2])
sns.scatterplot(x='Status', y='Total_Bad_Debt', data=df, ax=ax[1][0])
sns.scatterplot(x='Status', y='Total_Good_Debt', data=df, ax=ax[1][1])
sns.scatterplot(x='Status', y='Years_of_Working', data=df, ax=ax[1][2])
sns.scatterplot(x='Status', y='Owned_Realty', data=df, ax=ax[2][0])
sns.scatterplot(x='Status', y='Owned_Car', data=df, ax=ax[2][1])
sns.scatterplot(x='Status', y='Education_Type', data=df, ax=ax[2][2])


# In[18]:


corr_mat = df.corr(method='pearson')
  
# Retain upper triangular values of correlation matrix and
# make Lower triangular values Null
upper_corr_mat = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
  
# Convert to 1-D series and drop Null values
unique_corr_pairs = upper_corr_mat.unstack().dropna()
  
# Sort correlation pairs
sorted_mat = unique_corr_pairs.sort_values()
print(sorted_mat)


# In[19]:


df.info()


# In[25]:


X=df[['Applicant_Gender','Owned_Car','Owned_Realty','Total_Children','Total_Income','Income_Type','Education_Type','Family_Status',
    'Housing_Type','Owned_Mobile_Phone','Owned_Work_Phone','Owned_Phone','Owned_Email','Job_Title','Total_Family_Members','Applicant_Age',
    'Years_of_Working','Total_Bad_Debt','Total_Good_Debt']]
y=df[['Status']]


# In[26]:


X.shape


# In[27]:


y.shape


# In[29]:


# Export Features 
X.to_csv("X.csv")


# In[30]:


# Export dependent variable 
y.to_csv("y.csv")


# In[102]:


# X = df.iloc[:,1:-1] # X value contains all the variables except target variable('status' column)- df.iloc[row_start:row_end , col_start, col_end]
# y = df.iloc[:,-1:] # these are the labels


# In[28]:


# split data into train and test of its respective feature data columns and dependent columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=1)


# In[54]:


y_test.to_csv("y_test.csv")


# In[31]:


# we fit and transform the data into a scaler for accurate reading and results.
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_scaled = pd.DataFrame(mms.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(mms.transform(X_test), columns=X_test.columns)


# In[32]:


get_ipython().system('pip install imbalanced-learn')


# In[33]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_balanced, y_balanced = oversample.fit_resample(X_scaled, y_train)
X_test_balanced, y_test_balanced = oversample.fit_resample(X_test_scaled, y_test)


# In[34]:


y_train.value_counts()


# In[35]:


y_balanced.value_counts()


# In[59]:


y_test_balanced.to_csv("y_test_balanced.csv")


# In[60]:


train_balanced=pd.concat([X_balanced, y_balanced])


# In[62]:


train_balanced.to_csv("train_balanced.csv")


# In[63]:


test_balanced=pd.concat([X_test_balanced, y_test_balanced])


# In[64]:


test_balanced.to_csv("test_balanced.csv")


# In[36]:


y_balanced.shape


# In[37]:


y_balanced.head()


# In[38]:


y_test.value_counts()


# In[39]:


y_test_balanced.value_counts()


# In[40]:


pip install xgboost


# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[42]:


classifiers = {
    "LogisticRegression" : LogisticRegression(),
    "KNeighbors" : KNeighborsClassifier(),
    "SVC" : SVC(),
    "DecisionTree" : DecisionTreeClassifier(),
    "RandomForest" : RandomForestClassifier(),
    "XGBoost" : XGBClassifier()
}


# In[43]:


train_scores = []
test_scores = []

for key, classifier in classifiers.items():
    classifier.fit(X_balanced, y_balanced)
    train_score = classifier.score(X_balanced, y_balanced)
    train_scores.append(train_score)
    test_score = classifier.score(X_test_balanced, y_test_balanced)
    test_scores.append(test_score)

print(train_scores)
print(test_scores)


# In[44]:


xgb = XGBClassifier()
model = xgb.fit(X_balanced, y_balanced)
prediction = xgb.predict(X_test_balanced)


# In[45]:


from sklearn.metrics import classification_report
print(classification_report(y_test_balanced, prediction))


# In[82]:


# saving file for deployment
import pickle
with open('credit_model_xgboost.pkl', 'wb') as file:pickle.dump(classifier, file)


# In[ ]:




