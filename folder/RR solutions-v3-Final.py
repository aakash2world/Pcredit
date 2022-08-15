#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('C:\\Users\91939\Desktop\RR solutions')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("Application_Data.csv")


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


#we are filtering the columns that have non numeric values to see if they are useful
ot = pd.DataFrame(df.dtypes =='object').reset_index()
object_type = ot[ot[0] == True]['index']
object_type


# In[7]:


num_type = pd.DataFrame(df.dtypes != 'object').reset_index().rename(columns =  {0:'yes/no'})
num_type = num_type[num_type['yes/no'] ==True]['index']


# In[8]:


a = df[object_type]['Applicant_Gender'].value_counts()
b = df[object_type]['Income_Type'].value_counts()
c = df[object_type]['Education_Type'].value_counts()
d = df[object_type]['Family_Status'].value_counts()
e = df[object_type]['Housing_Type'].value_counts()
e = df[object_type]['Job_Title'].value_counts()

print( a,"\n",b,'\n', c, '\n', d, '\n', e)


# In[9]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for x in df:
    if df[x].dtypes=='object':
        df[x] = le.fit_transform(df[x])


# In[10]:


df.head(10)


# In[11]:


df.info()


# In[12]:


df[num_type].head()


# In[77]:


df.describe()


# In[13]:


df.corr()


# In[15]:


plt.figure(figsize = (16,10))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[24]:


corr_mat = df.corr(method='pearson')
  
# Retain upper triangular values of correlation matrix and
# make Lower triangular values Null
upper_corr_mat = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
  
# Convert to 1-D series and drop Null values
unique_corr_pairs = upper_corr_mat.unstack().dropna()
  
# Sort correlation pairs
sorted_mat = unique_corr_pairs.sort_values()
print(sorted_mat)


# In[102]:


X = df.iloc[:,:-1] # X value contains all the variables except target variable('status' column)- df.iloc[row_start:row_end , col_start, col_end]
y = df.iloc[:,-1:] # these are the labels


# In[103]:


X.shape


# In[104]:


y.shape


# In[105]:


# split data into train and test of its respective feature data columns and dependent columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


# In[106]:


# we fit and transform the data into a scaler for accurate reading and results.
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_scaled = pd.DataFrame(mms.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(mms.transform(X_test), columns=X_test.columns)


# In[28]:


get_ipython().system('pip install imbalanced-learn')


# In[107]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_balanced, y_balanced = oversample.fit_resample(X_scaled, y_train)
X_test_balanced, y_test_balanced = oversample.fit_resample(X_test_scaled, y_test)


# In[121]:


y_train.value_counts()


# In[123]:


y_balanced.value_counts()


# In[118]:


y_balanced.shape


# In[119]:


y_balanced.head()


# In[110]:


y_test.value_counts()


# In[111]:


y_test_balanced.value_counts()


# In[35]:


pip install xgboost


# In[112]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[113]:


classifiers = {
    "LogisticRegression" : LogisticRegression(),
    "KNeighbors" : KNeighborsClassifier(),
    "SVC" : SVC(),
    "DecisionTree" : DecisionTreeClassifier(),
    "RandomForest" : RandomForestClassifier(),
    "XGBoost" : XGBClassifier()
}


# In[114]:


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


# In[115]:


xgb = XGBClassifier()
model = xgb.fit(X_balanced, y_balanced)
prediction = xgb.predict(X_test_balanced)


# In[116]:


from sklearn.metrics import classification_report
print(classification_report(y_test_balanced, prediction))


# In[117]:


# saving file for deployment
import pickle
with open('credit_model_xgboost.pkl', 'wb') as file:pickle.dump(classifier, file)


# In[ ]:




