#!/usr/bin/env python
# coding: utf-8

# In[52]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[53]:


df=pd.read_csv("framingham.csv")
df


# In[54]:


x=df.drop('TenYearCHD',axis=1)
y=df["TenYearCHD"]


# In[55]:


x


# In[56]:


y


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


# In[58]:


model = LogisticRegression()


# In[59]:


model.fit(x_train,y_train)


# In[79]:


y_pred=model.predict(x_test)


# In[80]:


#fpr, tpr, thresh = roc_curve(y_test, pred_prob[:,1], pos_label=1)

# roc curve for tpr = fpr 

#random_probs = [0 for i in range(len(y_test))]
#p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[81]:


#auc_score = roc_auc_score(y_test, pred_prob[:,1])
#print("AUC score = ", auc_score)
from sklearn.metrics import accuracy_score
print('Accuracy =' ,accuracy_score(y_test,y_pred))


# In[82]:


plt.style.use('seaborn')


plt.plot(fpr, tpr, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

# title
plt.title('AUC_ROC curve')

# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# In[ ]:




