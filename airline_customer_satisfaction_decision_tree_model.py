#!/usr/bin/env python
# coding: utf-8

# In[201]:


import numpy as np 
import pandas as pd
import  matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[202]:


df = pd.read_csv('airline_data.csv')


# In[203]:


df.head()


# In[204]:


df.columns


# In[205]:


df.describe()


# # Drop the column ‘Unnamed’

# In[206]:


df = df.drop('Unnamed: 0',axis=1)


# In[207]:


df.head()


# # Replace all the “ “ in column with “_”

# In[208]:


df.columns=[each.replace(" ","_") for each in df.columns]


# In[209]:


df.columns


# # Give label to a satisfaction column value with out using any encoding method

# In[210]:


df["satisfaction"]=[1 if each=="satisfied" else 0 for each in df.satisfaction]


# # Plot the number of satisfied customers and the number of unsatisfied customers 

# In[211]:


df["satisfaction"].value_counts()


# In[212]:


df["satisfaction"].value_counts().plot(kind='bar')


# In[213]:


df["Gender"].value_counts().plot(kind='bar')


# # Find the mean value of satisfaction of male and female customers

# In[214]:


# Gender vs satisfaction
df[["Gender","satisfaction"]].groupby(["Gender"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)


# # Find the mean value of satisfaction of customers with respect to Age.
# 

# In[215]:


# Age vs satisfaction
df[["Age","satisfaction"]].groupby(["Age"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)


# # Find the mean value of satisfaction of customers with respect to Food_and_drink.

# In[216]:


#Food_and_drink vs satisfaction
df[["Food_and_drink","satisfaction"]].groupby(["Food_and_drink"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)


# # Display a boxplot for Flight_Distance

# In[217]:


df['Flight_Distance'].plot(kind = 'box')


# # Display a boxplot for Checkin_service
# 

# In[218]:


df['Checkin_service'].plot(kind = 'box')


# # Find all the Null values 

# In[219]:


df.isnull().sum()


# # Drop all the na values

# In[220]:


df = df.dropna()


# In[221]:


df.isnull().sum()


# # Find the unique values in Flight_Distance 

# In[222]:


df['Flight_Distance'].unique()


# In[ ]:





# # Part-2: Working with Models 

# # 1) Perform encoding in columns Gender, Customer_Type, Type_of_Travel, and Class. 

# In[ ]:





# In[223]:


from sklearn.preprocessing import LabelEncoder


# In[224]:


enc = LabelEncoder()


# In[225]:


df['Gender'] = enc.fit_transform(df['Gender'])


# In[226]:


df['Customer_Type'] = enc.fit_transform(df['Customer_Type'])
df['Type_of_Travel'] = enc.fit_transform(df['Type_of_Travel'])
df['Class'] = enc.fit_transform(df['Class'])


# In[227]:


df.info()


# # 2) Drop the column id & Unnamed:_0.1

# In[228]:


df = df.drop(['id','Unnamed:_0.1'],axis=1)


# In[229]:


df.head()


# # 3) Create the features and target Data

# In[230]:


X = df.drop('satisfaction',axis=1)
y = df['satisfaction']


# In[231]:


X.shape


# # 4) Perform scaling on features data 

# In[232]:


from sklearn.preprocessing import StandardScaler


# In[233]:


# Normalize Features
scaler = StandardScaler()
X_scal = scaler.fit_transform(X)


# In[234]:


from sklearn.model_selection import train_test_split


# # 5) Split the data in training and testing sets 

# In[235]:


X_train,X_test,y_train,y_test = train_test_split(X_scal,y,test_size=0.2)


# In[236]:


X_train.shape


# In[237]:


X_test.shape


# # 6) Fit the decision tree model with various parameters 

# In[238]:


from sklearn.tree import DecisionTreeClassifier
params_dt = {'max_depth': 12,    
             'max_features': "sqrt",
            'min_samples_leaf':1,
             'min_samples_split':2,
            'criterion':'gini'}

model_dt = DecisionTreeClassifier(**params_dt)


# In[239]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score


# # 7) Create a function to display precision score, recall score, accuracy, classification report, confusion matrix, F1 Score

# In[240]:


def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train,y_train.ravel())
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("pricison_score: ",precision_score(y_test, y_pred))
    print("recall_score: ",recall_score(y_test, y_pred))
    print("Accuracy = {}".format(accuracy))
    print(classification_report(y_test,y_pred,digits=5))
    print(confusion_matrix(y_test,y_pred))
    


# In[241]:


run_model(model_rf,X_train, y_train, X_test, y_test)

