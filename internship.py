#!/usr/bin/env python
# coding: utf-8

# # Molecular Biology (Promoter Gene Sequences)

# ### Import All Required Libraries

# In[1]:


# load libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)


# ### Import dataset

# In[2]:


names = ['Class', 'id', 'Sequence']
data=pd.read_csv('promoters.data',delimiter=',',names=names)


# In[3]:


#View First 5 rows of the data
data.head()


# ### Pre Processing the data

# The data is not in a usable form; as a result, we will need to process it before using it to train our algorithms.

# In[4]:


data.head()


# In[ ]:





# In[5]:


# Building our Dataset by creating a custom Pandas DataFrame
# Each column in a DataFrame is called a Series. Lets start by making a series for each column.

classes = data.loc[:, 'Class']
print(classes[:5])


# In[6]:


# generate list of DNA sequences
import sys
sequences = list(data.loc[:, 'Sequence'])
dataset = {}

# loop through sequences and split into individual nucleotides
for i, seq in enumerate(sequences):
    
    # split into nucleotides, remove tab characters
    nucleotides = list(seq)
    nucleotides = [x for x in nucleotides if x != '\t']
    
    # append class assignment
    nucleotides.append(classes[i])
    
    # add to dataset
    dataset[i] = nucleotides
    
print(dataset[0])


# In[7]:


# turn dataset into pandas DataFrame
dframe = pd.DataFrame(dataset)
print(dframe)


# In[8]:


df = dframe.transpose()
df.head()
# transpose the DataFrame


# In[9]:


# for clarity, lets rename the last dataframe column to class
df.rename(columns = {57: 'Class'}, inplace = True) 
df.head()


# In[10]:


df.columns = ['p-50','p-49','p-48','p-47','p-46','p-45','p-44','p-43','p-42','p-41','p-40','p-39','p-38','p-37','p-36','p-35',
              'p-34','p-33','p-32','p-31','p-30','p-29','p-28','p-27','p-26','p-25','p-24','p-23','p-22','p-21','p-21','p-19',
             'p-18','p-17','p-16','p-15','p-14','p-13','p-12','p-11','p-10','p-9','p-8','p-7','p-6','p-5','p-4','p-3','p-2',
             'p-1','p1','p2','p3','p4','p5','p6','p7','Class']


# In[11]:


df.head()


# In[12]:


df.shape


# In[13]:


df.info()


# In[14]:


#To describe object type features of train data
df.describe(include=['object'])


# In[15]:


#observe the target variable
df.groupby('Class').size() 


# In[16]:


#Finding missing values
df.isnull().sum()


# In[17]:


plt.figure(figsize=(15,5))
sns.heatmap(df.isnull(), cbar = False, yticklabels=False, cmap="inferno" )


# In[18]:


#finding outliers
#Since all the data is categorical data then there are no outliers


# # EDA

# In[19]:


fig, ax=plt.subplots(3,3,figsize=(26,15))
ax[0,0].bar(df['p-50'].value_counts().index,df['p-50'].value_counts())
ax[0,1].bar(df['p-49'].value_counts().index,df['p-49'].value_counts())
ax[0,2].bar(df['p-48'].value_counts().index,df['p-48'].value_counts())
ax[1,0].bar(df['p-47'].value_counts().index,df['p-47'].value_counts())
ax[1,1].bar(df['p-46'].value_counts().index,df['p-46'].value_counts())
ax[1,2].bar(df['p-45'].value_counts().index,df['p-45'].value_counts())
ax[2,0].bar(df['p-44'].value_counts().index,df['p-44'].value_counts())
ax[2,1].bar(df['p-43'].value_counts().index,df['p-43'].value_counts())
ax[2,2].bar(df['p-42'].value_counts().index,df['p-42'].value_counts())

plt.draw()
ax[0,1].set_xticklabels(ax[0,1].get_xticklabels(), rotation=45)


# In[20]:


fig, ax=plt.subplots(3,3,figsize=(26,15))
ax[0,0].bar(df['p-41'].value_counts().index,df['p-41'].value_counts())
ax[0,1].bar(df['p-40'].value_counts().index,df['p-40'].value_counts())
ax[0,2].bar(df['p-39'].value_counts().index,df['p-39'].value_counts())
ax[1,0].bar(df['p-38'].value_counts().index,df['p-38'].value_counts())
ax[1,1].bar(df['p-37'].value_counts().index,df['p-37'].value_counts())
ax[1,2].bar(df['p-36'].value_counts().index,df['p-36'].value_counts())
ax[2,0].bar(df['p-35'].value_counts().index,df['p-35'].value_counts())
ax[2,1].bar(df['p-34'].value_counts().index,df['p-34'].value_counts())
ax[2,2].bar(df['p-33'].value_counts().index,df['p-33'].value_counts())

plt.draw()
ax[0,1].set_xticklabels(ax[0,1].get_xticklabels(), rotation=45)


# In[ ]:





# In[ ]:





# In[21]:


# we can't run machine learning algorithms on the data in 'String' formats. As a result, we need to switch
# it to numerical data. This can easily be accomplished using the pd.get_dummies() function
numerical_df = pd.get_dummies(df)
numerical_df.head()


# In[22]:


numerical_df.shape


# In[23]:


# We don't need both class columns.  Lets drop one then rename the other to simply 'Class'.
df = numerical_df.drop(columns=['Class_-'])

df.rename(columns = {'Class_+': 'Class'}, inplace = True)
df.shape


# In[24]:


# generate count statistics of duplicate entries
if len(df[df.duplicated()]) > 0:
    print("No. of duplicated entries: ", len(df[df.duplicated()]))
    print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head())
else:
    print("No duplicated entries found")


# ### separate training and testing datasets

# In[25]:


from sklearn import model_selection

# Create X and Y datasets for training
X = df.drop(['Class'], 1)
y = df['Class']


# split data into training and testing datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=10)


# In[26]:


X.head()


# In[27]:


y.head()


# In[28]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Training and Testing the Classification Algorithms

# ### Logistic Regression

# In[29]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)


# In[30]:


y_pred=logreg.predict(X_test)


# In[31]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[32]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[33]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# ### Logistic regression ROC curve

# In[34]:


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# ### Random Forest

# In[35]:


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(X_train, y_train)


# In[36]:


y_pred=model.predict(X_test)


# In[37]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[38]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# ### Naive bayes Theorem

# In[39]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#fit on training data
gnb.fit(X_train, y_train)


# In[40]:


y_pred=model.predict(X_test)


# In[41]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[42]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# ### SVM

# In[43]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


# In[44]:


y_pred = svclassifier.predict(X_test)


# In[45]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# ### Polinomeal Kernal

# In[46]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)


# In[47]:


y_pred = svclassifier.predict(X_test)


# In[48]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ### Gaussian Kernel

# In[49]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)


# In[50]:


y_pred = svclassifier.predict(X_test)


# In[51]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ### Sigmoid Kernel

# In[52]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)


# In[53]:


y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[54]:


import pickle
#open a file where you want to store your data
file=open('DNA_promoter.pk1','wb')

#dump information to thet file
pickle.dump(logreg,file)


# In[ ]:




