#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


url="https://raw.githubusercontent.com/sumeyyeozel/csv/main/Groceries_dataset.csv"
groceries=pd.read_csv(url)


# In[3]:


groceries.head()


# In[4]:


groceries.shape


# In[5]:


groceries.dtypes


# In[6]:


groceries.describe()


# In[7]:


groceries.columns


# In[8]:


groceries.index


# In[9]:


groceries['Date']=pd.to_datetime(groceries['Date'])


# In[10]:


groceries.dtypes


# In[11]:


groceries.isna().sum()


# In[12]:


groceries.isnull().sum()


# In[13]:


groceries['Member_number'].value_counts()


# In[14]:


groceries['Date'].value_counts()


# In[15]:


groceries['itemDescription'].value_counts()


# In[16]:


groceries.groupby('itemDescription').mean()


# In[57]:


topten = groceries['itemDescription'].value_counts().sort_values(ascending=False)[:10]
fig = px.bar(x= topten.index, y= topten.values,
            color=groceries['itemDescription'].value_counts().sort_values(ascending=False)[:10])
fig.update_layout(title_text= "Top 10 frequently sold products ", xaxis_title= "Products", yaxis_title="Number of item sold")
fig.show()


# In[18]:


leastten = groceries['itemDescription'].value_counts().sort_values(ascending=True)[:10]
fig = px.bar(x= leastten.index, y= leastten.values, 
             color=groceries['itemDescription'].value_counts().sort_values(ascending=True)[:10])
fig.update_layout(title_text= "Least 10 frequently sold products ", xaxis_title= "Products", yaxis_title="Number of item sold")
fig.show()


# In[19]:


pd.DataFrame(groceries['Member_number'].value_counts().sort_values(ascending=False))[:10]


# find the dates on which highest sale was made.

# In[20]:


fig1 = px.bar(groceries["Date"].value_counts(ascending=False), 
              orientation= "v", 
              color = groceries["Date"].value_counts(ascending=False),
              
               labels={'value':'Count', 'index':'Date','color':'Meter'})

fig1.update_layout(title_text="Exploring highest sales by  date")

fig1.show()


# In[21]:


import datetime as dt
newdate = groceries['Date'].dt.strftime('%Y-%m')
print(newdate) 


# In[80]:


fig2 = px.bar(newdate.value_counts(ascending=False), 
              orientation= "v", 
              color = newdate.value_counts(ascending=False),
              
               labels={'value':'Count', 'index':'Date','color':'Meter'})

fig2.update_layout(title_text="Exploring highest sales by  date")

fig2.show()


# In[22]:


products=groceries['itemDescription'].unique()


# In[23]:


products[:10]


# One Hot Encoder

# In[24]:


one_hot = pd.get_dummies(groceries['itemDescription'])
groceries1=groceries.copy()
groceries1.drop(['itemDescription'], inplace =True, axis=1)

groceries1 = groceries1.join(one_hot)

groceries1.head()


# In[25]:


groceries2 = groceries1.groupby(['Member_number', 'Date'])[products[:]].sum()

groceries2.head()


# In[26]:


#Reset the index of the newly formed dataset.
groceries2 = groceries2.reset_index()[products]
groceries2.head()


# In[27]:


def product_names(x):
    for product in products:
        if x[product] >0:
            x[product] = product
    return x
#Apply the created function on data2 dataset.
groceries2 = groceries2.apply(product_names, axis=1)
groceries2.head()


# The bottom line is required for apriori algorithm.

# In[28]:


#Filter out the values from the groceries frame groceries2
x = groceries2.values
#Convert into list values in each row if value is not zero
x = [sub[~(sub==0)].tolist() for sub in x if sub [sub != 0].tolist()]
transactions = x
transactions[0:10] 


# Apriori Algorithm
# Apriori is an algorithm for frequent itemset mining and association rule learning over relational databases. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the database. The frequent itemsets determined by Apriori can be used to determine association rules which highlight general trends in the database: this has applications in domains such as market basket analysis.

# In[31]:


get_ipython().system('pip install apyori')


# In[32]:


import apyori
from apyori import apriori


# Association rules is used to find relationships between attributes in large databases. An association rule, A=> B, will be of the form” for a set of transactions, some value of itemset A determines the values of itemset B under the condition in which minimum support and confidence are met”.

# In[55]:


associations = apriori(transactions, min_support = 0.00030, min_confidence = 0.06, min_lift = 3, max_length = 2, target = "associations")
association_results = list(associations)
association_results[:5]


# In[51]:


#iterate through the list of associations and for each item
for item in association_results:
    
    #for each item filter out the item pair and create item list containing individual items in the itemset
    itemset = item[0]
    items = [x for x in itemset]
    
    #Print the relationship( First value in items to second value in items)
    print("Rule : ", items[0], " -> " + items[1])
    
    #Print support,confidence and lift value of each itemset
    print("Support : ", str(item[1]))
    print("Confidence : ",str(item[2][0][2]))
    print("Lift : ", str(item[2][0][3]))
    print("===================") 


# In[ ]:




