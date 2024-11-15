#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[14]:


df = pd.read_csv('diabetes.csv')


# In[15]:


df.head()


# In[17]:


df['Outcome'].value_counts()


# In[18]:


x=df.drop('Outcome',axis=1)
y=df.Outcome


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)


# # upsampling

# In[21]:


from sklearn.utils import resample


# In[22]:


x=pd.concat([x_train,y_train],axis=1)


# In[25]:


not_dia= x[x.Outcome==0]
diabetic= x[x.Outcome==1]


# In[32]:


#upscaling minority
dia_upsampled = resample(diabetic,
                        replace=True,
                        n_samples=len(not_dia),
                        random_state=27)


# In[34]:


upsampled=pd.concat([not_dia,dia_upsampled])


# In[36]:


upsampled.Outcome.value_counts()


# # Down Sampling

# In[62]:


not_dia_downsampled = resample(not_dia,
                        replace=False,
                        n_samples=len(diabetic),
                        random_state=27)


# In[63]:


downsampled = pd.concat([not_dia_downsampled, diabetic])


# In[64]:


downsampled.Outcome.value_counts()


# In[ ]:




