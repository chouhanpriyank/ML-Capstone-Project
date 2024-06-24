#!/usr/bin/env python
# coding: utf-8

# # USING DENTAL METRICS TO PREDICT GENDER
Predicting gender based on dental metrics can be approached as a classification problem in machine learning. 
# # Data Collection and Preprocessing
# 
Data preprocessing is a critical step to ensure the quality and reliability of the dataset. This process includes

1. Data Cleaning: Handling missing values, correcting inconsistencies, and removing any outliers that    could skew the analysis.

2. Feature Engineering: Creating new features from existing data to enhance the model's predictive        power. For example, calculating ratios or differences between certain measurements.

3. Normalization and Scaling: Standardizing the data to ensure that features with different scales do    not disproportionately influence the model.

# # Exploratory Data Analysis (EDA)
# 
1. Visualize Data: Explore the distribution of each feature with respect to gender.

2. Correlation Analysis: Examine correlations between features and between features and the target        variable (gender).

# # Model Development
# 
The core of the project is the development of a machine learning model to predict gender based on dental features. Several algorithms are considered, including:

1. Logistic Regression
2. Decision Trees
3. Random Forests
4. XG Boost Classifier

The dataset is split into training and testing sets to evaluate the models' performance. Cross-validation is employed to ensure that the model generalizes well to unseen data.
# # Model Evaluation and Interpretation
# 
The performance of the predictive models is assessed using accuracy metrics. Confusion matrices provide additional insights into the models' ability to distinguish between genders.

# # Conclusion
# 
The project aims to establish a robust methodology for predicting gender based on dental features, contributing to both academic research and practical applications in fields such as forensic science and archaeology. Future work may involve expanding the dataset to include a more diverse population, exploring additional dental features, and refining the model to improve its accuracy and generalizability.

By integrating dental anthropology with advanced data science techniques, this project paves the way for innovative approaches to understanding the complex interplay between biology and gender, ultimately enhancing our ability to make accurate and meaningful predictions from dental data.


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
     


# In[2]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


# In[3]:


from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix


# In[4]:


data = pd.read_csv("Dentistry Dataset.csv")


# In[5]:


data.head(20)


# In[6]:


#Data Preprocessing
#checking null values
data.isnull().sum()


# In[7]:


data.describe()


# In[8]:


#visualizing intercanine distance data
#sns.lineplot(data['inter canine distance intraoral'], data['intercanine distance casts'], hue=data["Gender"])
sns.lineplot(x='inter canine distance intraoral', y='intercanine distance casts', hue='Gender', data=data)


# In[9]:


sns.lineplot(x='right canine width intraoral', y='right canine width casts', hue='Gender', data=data)


# In[10]:


sns.lineplot(x='left canine width intraoral', y='left canine width casts', hue='Gender', data=data)


# In[11]:


sns.lineplot(x='right canine index intra oral', y='right canine index casts', hue='Gender', data=data)


# In[12]:


sns.lineplot(x='left canine index intraoral', y='left canine index casts', hue='Gender', data=data)


# In[13]:


#encoding the data of Gender from categorical to numerical
twogender = {'Female':0, 'Male':1}
data['Gender'] = data['Gender'].map(twogender)


# In[14]:


data.head()


# In[15]:


X = data.drop(['Gender','Sample ID'], axis=1)
y = data['Gender']


# In[16]:


X


# In[17]:


y


# In[18]:


#splitting the data into training and testing set
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)


# In[19]:


print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)


# # Trying Diffrent Methods to find the best accuracy
# 

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[21]:


#Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_val)
dt_acc=accuracy_score(y_val, y_pred)
#print(accuracy_score(y_val, y_pred))
print('Accuracy of Decision Tree is: {:.2f}%'.format(dt_acc*100))



# In[22]:


#random forest Classifier
rf = RandomForestClassifier(random_state =0)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_acc = accuracy_score(y_val, rf_pred)
print('Accuracy of Random Forest is: {:.2f}%'.format(rf_acc*100))


# In[23]:


#logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_val)
lr_acc = accuracy_score(y_val, lr_pred)
print('Accuracy of Logistic Regression is: {:.2f}%'.format(lr_acc*100))



# In[24]:


#Gradient Boosting Classifier
clf = GradientBoostingClassifier(random_state=0)
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_val)
clf_acc = accuracy_score(y_val, clf_pred)
print('Accuracy of Gradient Boosting Classifier is: {:.2f}%'.format(clf_acc*100))


# # Accuracy of Decision Tree is: 100.00%

# # Accuracy of Random Forest is: 98.18%
# 

# # Accuracy of Logistic Regression is: 90.91%
# 

# # Accuracy of Gradient Boosting Classifier is: 100.00%

# # Trying some different method

# In[25]:


plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[ ]:




