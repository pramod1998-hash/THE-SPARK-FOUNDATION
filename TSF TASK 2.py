#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing librabries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#importing data to get read
data = pd.read_csv('http://bit.ly/w-data')
data.head()


# In[5]:


#understanding the data
data.describe()


# In[6]:


data.info()


# In[8]:


x = data.iloc[:, :-1].values
y = data.iloc[:,1].values
x
y


# In[9]:


#spitting of data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,random_state = 0)


# In[10]:


#applying algorithm
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)


# In[11]:


line= regression.coef_*x+regression.intercept_


# In[13]:


#scatter ploting
plt.scatter(x, y, label= "Data points", color= "green", marker= "*", s=30)
plt.plot(x,line)
plt.show()


# In[14]:


#marking prediction
print(x_test)
y_pred = regression.predict(x_test)
y_pred


# In[16]:


plt.scatter(x_train,y_train,color = 'blue')
plt.plot(x_test,y_pred, color = 'black')
plt.xlabel('No. of study hours')
plt.ylabel('Percentage of score')


# In[17]:


#Actual vs predicted
ds_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
ds_new


# In[18]:


#final prediction of a student for 9.5 hours of study
Hours = 9.5
print('No.of hours studied by the student =', Hours)
print('Predicted value of score =',regression.predict(np.array(Hours).reshape(1,-1))[0])


# In[ ]:




