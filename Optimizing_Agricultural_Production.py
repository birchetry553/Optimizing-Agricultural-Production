#!/usr/bin/env python
# coding: utf-8

# # Optimizing Agricultural Production

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for interactivity
from ipywidgets import interact


# In[14]:


import os 
os.chdir("C:/Users/dell/Documents/datasets/data")


# In[15]:


data=pd.read_csv("data.csv")


# In[16]:


data.head()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data['label'].value_counts()


# In[17]:


# let us check the summary for all the crops

print("Average ratio of Nitrogen in the soil : {0:.2f}".format(data['N'].mean()))
print("Average ratio of Phosphorus in the soil : {0:.2f}".format(data['P'].mean()))
print("Average ratio of Potassium in the soil : {0:.2f}".format(data['K'].mean()))
print("Average Temperature in Celsius : {0:.2f}".format(data['temperature'].mean()))
print("Average Relative Humidity in % : {0:.2f}".format(data['humidity'].mean()))
print("Average PH value of the soil : {0:.2f}".format(data['ph'].mean()))
print("Average Rainfall in mm : {0:.2f}".format(data['rainfall'].mean()))


# In[18]:


# check the summary statistics for each of the crops

@interact
def summary(crops=list(data['label'].value_counts().index)):
    x=data[data['label']==crops]
    print("-------------------------------------------------")
    print("Statistics for Nitrogen")
    print("Minimum Nitrogen required :",x['N'].min())
    print("Average Nitrogen required :",x['N'].mean())
    print("Maximum Nitrogen required :",x['N'].max())
    print("--------------------------------------------------")
    print("Statistics for phosphorous")
    print("Minimum Phosphorous required :",x['P'].min())
    print("Average Phosphorous required :",x['P'].mean())
    print("Maximum Phosphorous required :",x['P'].max())
    print("--------------------------------------------------")
    print("Statistics for potassium")
    print("Minimum potassium required :",x['K'].min())
    print("Average potassium required :",x['K'].mean())
    print("Maximum potassium required :",x['K'].max())
    print("---------------------------------------------------")
    print("Statistics for Temperature")
    print("Minimum Temperature required :",x['temperature'].min())
    print("Average Temperature required :",x['temperature'].mean())
    print("Maximum Temperature required :",x['temperature'].max())
    print("-----------------------------------------------------")
    print("Statistics for Relative Humidity")
    print("Minimum Humidity required :",x['humidity'].min())
    print("Average Humidity required :",x['humidity'].mean())
    print("Maximum Humidity required :",x['humidity'].max())
    print("------------------------------------------------------")
    print("Statistics for PH value")
    print("Minimum PH value required :",x['ph'].min())
    print("Average PH value required :",x['ph'].mean())
    print("Maximum PH value required :",x['ph'].max())
    print("-------------------------------------------------------")
    print("Statistics for Rainfall")
    print("Minimum Rainfall required :",x['rainfall'].min())
    print("Average Rainfall required :",x['rainfall'].mean())
    print("Maximum Ra  infall required :",x['rainfall'].max())
    


# In[12]:


# lets compare the average requirements for each crops with average condition

@interact
def compare(conditions=['N','P','K','temperature','ph','humidity','rainfall']):
    print("Average value for", conditions,"is {0:.2f}".format(data[conditions].mean()))
    print("------------------------------------------------------")
    print("Rice : {0:.2f}".format(data[(data['label']=='rice')][conditions].mean()))
    print("Black Grams : {0:.2f}".format(data[(data['label']=='blackgram')][conditions].mean()))
    print("Banana : {0:.2f}".format(data[(data['label']=='banana')][conditions].mean()))
    print("Jute : {0:.2f}".format(data[(data['label']=='jute')][conditions].mean()))
    print("Coconut : {0:.2f}".format(data[(data['label']=='coconut')][conditions].mean()))
    print("Apple : {0:.2f}".format(data[(data['label']=='apple')][conditions].mean()))
    print("Papaya : {0:.2f}".format(data[(data['label']=='papaya')][conditions].mean()))
    print("Muskmelon : {0:.2f}".format(data[(data['label']=='muskmelon')][conditions].mean()))
    print("Grapes : {0:.2f}".format(data[(data['label']=='grapes')][conditions].mean()))
    print("Watermelon : {0:.2f}".format(data[(data['label']=='watermelon')][conditions].mean()))
    print("Kidney Beans : {0:.2f}".format(data[(data['label']=='kidneybeans')][conditions].mean()))
    print("Mung Beans : {0:.2f}".format(data[(data['label']=='mungbean')][conditions].mean()))
    print("Oranges : {0:.2f}".format(data[(data['label']=='orange')][conditions].mean()))
    print("Cheak Peas : {0:.2f}".format(data[(data['label']=='chickpea')][conditions].mean()))
    print("Lentils : {0:.2f}".format(data[(data['label']=='lentil')][conditions].mean()))
    print("Cotton : {0:.2f}".format(data[(data['label']=='cotton')][conditions].mean()))
    print("Maize : {0:.2f}".format(data[(data['label']=='maize')][conditions].mean()))
    print("Moth Beans : {0:.2f}".format(data[(data['label']=='mothbeans')][conditions].mean()))
    print("Pigeon Peas : {0:.2f}".format(data[(data['label']=='pigeonpeas')][conditions].mean()))
    print("Mango : {0:.2f}".format(data[(data['label']=='mango')][conditions].mean()))
    print("Pomegranate: {0:.2f}".format(data[(data['label']=='pomegranate')][conditions].mean()))
    print("Coffee : {0:.2f}".format(data[(data['label']=='coffee')][conditions].mean()))
    
    
    


# In[19]:


# let make this function more intuitive

@interact
def compare(conditions=['N','P','K','temperature','ph','humidity','rainfall']):
    print("Crops which require greater than average",conditions,'\n')
    print(data[data[conditions]> data[conditions].mean()]['label'].unique())
    print("----------------------------------------------------------------")
    print("Crops which require less than average",conditions,'\n')
    print(data[data[conditions]<data[conditions].mean()]['label'].unique())


# In[30]:


plt.figure(figsize=(20,9))
plt.subplot(2,4,1)
sns.distplot(data['N'],color="darkblue")
plt.xlabel("Ratio of Nitrogen", fontsize=12)
plt.grid()



plt.subplot(2,4,2)
sns.distplot(data['P'],color="blue")
plt.xlabel("Ratio of Phosphorous", fontsize=12)
plt.grid()



plt.subplot(2,4,3)
sns.distplot(data['K'],color="black")
plt.xlabel("Ratio of Potassium", fontsize=12)
plt.grid()




plt.subplot(2,4,4)
sns.distplot(data['temperature'],color="green")
plt.xlabel("Ratio of Temperature", fontsize=12)
plt.grid()




plt.subplot(2,4,5)
sns.distplot(data['ph'],color="red")
plt.xlabel("Ratio of PH", fontsize=12)
plt.grid()




plt.subplot(2,4,6)
sns.distplot(data['humidity'],color="yellow")
plt.xlabel("Ratio of Humidity", fontsize=12)
plt.grid()




plt.subplot(2,4,7)
sns.distplot(data['rainfall'],color="pink")
plt.xlabel("Ratio of Rainfall", fontsize=12)
plt.grid()



plt.suptitle("Distribution of Agricultural Conditions",fontsize=22)
plt.show()


# In[31]:


# lets find out some interesting facts

print("Some Interesting Patterns")
print("---------------------------------------------")
print("Crops which require very high ratio of Nitrogen content in Soil:",data[data['N']>120]['label'].unique())
print("Crops which require very high ratio of Phosphorous content in Soil:",data[data['P']>100]['label'].unique())
print("Crops which require very high ratio of Potassium content in Soil:",data[data['K']>200]['label'].unique())
print("Crops which require very high Rainfall:",data[data['rainfall']>200]['label'].unique())
print("Crops which require very low Temperature:",data[data['temperature']<10]['label'].unique())
print("Crops which require very high Temperature:",data[data['temperature']>40]['label'].unique())
print("Crops which require very Low Humidity:",data[data['humidity']<20]['label'].unique())
print("Crops which require very Low PH:",data[data['ph']<4]['label'].unique())
print("Crops which require very high PH:",data[data['ph']>9]['label'].unique())


# In[32]:


# lets understand which crops can only be grown in Summer Season, Winter Season and Rainy Season

print("Summer Crops")
print(data[(data['temperature']>30) & (data['humidity']>50)]['label'].unique())
print("-----------------------------------------------------------")
print("Winter Crops")
print(data[(data['temperature']<20) & (data['humidity']>30)]['label'].unique())
print("------------------------------------------------------------")
print("Rainy Crops")
print(data[(data['rainfall']>200) & (data['humidity']>30)]['label'].unique())


# In[33]:


from sklearn.cluster import KMeans

# removing the label columns
x=data.drop(['label'],axis=1)

# selecting all the values of the data
x=x.values

# checking the shape
print(x.shape)


# In[40]:


# Lets determine the optimum number of Clusters within the Dataset:

plt.rcParams['figure.figsize']==(10,4)

wcss=[]
for i in range (1,11):
    km=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)
    
# lets plot the results
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method",fontsize=20)
plt.xlabel("No of Clusters")
plt.ylabel("wcss")
plt.show()


# In[41]:


# lets implement the K-means algorithm to perform clustering analysis

km=KMeans(n_clusters=4, init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means=km.fit_predict(x)

# lets find out the results:
a=data['label']
y_means=pd.DataFrame(y_means)
z=pd.concat([y_means,a],axis=1)
z=z.rename(columns={0:'cluster'})

# lets check the clusters of each crops

print("Lets Check The Result after applying the K-means Algorithm: \n")
print("Crops in First Cluster:",z[z['cluster']==0]['label'].unique())
print("---------------------------------------------------------------")
print("Crops In Second Cluster:",z[z['cluster']==1]['label'].unique())
print("----------------------------------------------------------------")
print("Crops in Third Cluster:",z[z['cluster']==2]['label'].unique())
print("-----------------------------------------------------------------")
print("Crops in Fourth Cluster:",z[z['cluster']==3]['label'].unique())


# # we use Logistic Regression Model

# In[42]:


# lets split the dataset for predictive modelling

y=data['label']
x=data.drop(['label'],axis=1)
print("shape of x:",x.shape)
print("shape of y:",y.shape)


# In[43]:


# lets create Training and Testing Sets for validation of results.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print("the shape of x train:",x_train.shape)
print("the shape of x test:",x_test.shape)
print("the shape of y train:",y_train.shape)
print("the shape of y test:",y_test.shape)


# In[44]:


# LETS CREATE A PREDICTIVE MODEL

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[45]:


# Lets evaluate the model performance

from sklearn.metrics import confusion_matrix

# lets print the confusion matrix first

plt.rcParams['figure.figsize']=(10,10)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,cmap='Wistia')
plt.title("Confusion matrix for Logistic Regression", fontsize=15)
plt.show()


# In[51]:


# lets print the classification report of model

from sklearn.metrics import classification_report
cr= classification_report(y_test,y_pred)
print(cr)


# In[52]:


# lets check the head of the dataset

data.head()


# In[53]:


prediction=model.predict((np.array([[90,
                                     40,
                                     40,
                                     20,
                                     80,
                                     7,
                                     200]])))

print("The suggested crop for given Climatic Condition is: ",prediction)


# In[55]:


prediction=model.predict((np.array([[92,
                                     20,
                                     39,
                                     12,
                                     56,
                                     9,
                                     111]])))

print("the suggested crop for given climatic condition is:",prediction)


# In[ ]:




