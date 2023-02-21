#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")


# In[2]:


ipl = pd.read_csv('IPL_Matches_2008-2020.csv')


# In[3]:


ipl.head()
ipl.columns


# In[4]:


ipl["team1"].value_counts()


# In[5]:


ipl["city"].value_counts()


# In[6]:


ipl["toss_decision"].value_counts()


# In[7]:


#for Delhi Capitals
ipl['team1']=ipl['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
ipl['team2']=ipl['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
ipl['winner']=ipl['winner'].str.replace('Delhi Daredevils','Delhi Capitals')
ipl['toss_winner']=ipl['toss_winner'].str.replace('Delhi Daredevils','Delhi Capitals')

#ipl = ipl.drop(['neutral_venue'], axis=1)


# In[8]:


index_names = ipl[(ipl.team1 == 'Deccan Chargers') | (ipl.team2 == 'Deccan Chargers') | (ipl.winner == 'Deccan Chargers') | (ipl.toss_winner == 'Deccan Chargers')].index
ipl.drop(index_names, inplace = True)


# In[9]:


ipl.dtypes


# In[10]:


#converting Date from string to "date time" format
#ipl['date'] = pd.to_datetime(ipl['date'])
#ipl.dtypes


# Now we make **Predictors** columns for the target variable:

# In[11]:


ipl.drop(["date","id", "player_of_match", 'umpire1', "venue", "umpire2","result","result_margin","eliminator","method"], axis=1, inplace=True)


# In[12]:


print(ipl.head())


# In[13]:


X = ipl.drop(["winner"], axis=1)
y = ipl["winner"]


# In[14]:


X = pd.get_dummies(X, ["city", "team1","team2", "toss_winner", "toss_decision"], drop_first = True)


# In[ ]:


print(X.columns)


# In[15]:


ipl.head()


# In[16]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

ipl.dtypes


# In[17]:


ipl.columns


# In[46]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)


# In[69]:


from sklearn.ensemble import RandomForestClassifier
model_ipl = RandomForestClassifier(n_estimators=200,criterion='gini',max_depth=6,min_samples_leaf= 2,max_features='auto')


# 
# Now we use GridSearchCV to **Tune our Hyper parameters**:
# We first set the params that we want to tune in a Dictionary
# 

# In[62]:


''''model = RandomForestClassifier(random_state=1)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy'],
    'min_samples_leaf':[1,2,3]
}'''


# In[63]:


#from sklearn.model_selection import GridSearchCV


# In[64]:


#to check for all the available params in the model:
#model.get_params().keys()


# In[65]:


#model_ipl = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
#model_ipl.fit(x_train, y_train)


# Outputs the best possible options within the given parameters in the dictionary:

# In[66]:


#model_ipl.best_params_


# Here we *train* the model:

# In[70]:


model_ipl.fit(x_train, y_train)


# Now we **Test** the model:

# In[77]:


y_pred = model_ipl.predict(x_test)


# In[78]:





# **Note:** 
# 
# -> Higher the estimators, longer runtime but more accurate
# 
# -> min_samples_split: number of samples in each leaf, higher means lesser overfit but lower accuracy
# 

# In[79]:


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_pred, y_test)
ac




# In[80]:


#print("--> Enter City, Neutral Venue (0 for False and 1 for True), Team1, Team2, Name of the Team that Won Toss, Toss Decision")
#print("--> Note: Enter team names as given below and note captilisations \n")
#print("Royal Challengers Bangalore\nMumbai Indians\nKolkata Knight Riders\nChennai Super Kings\nKings XI Punjab\nDelhi Daredevils\nRajasthan Royals\nSunrisers Hyderabad \nPune Warriors\nGujarat Lions\nKochi Tuskers Kerala\nRising Pune Supergiants\nRising Pune Supergiant")


# In[117]:


city = input()
neutral = int(input())
team1 = input()
team2 = input()
t_winner = input()
t_decision = input()


# In[118]:


user_input = pd.DataFrame(data = [[city,neutral,team1,team2,t_winner, t_decision]], columns = ['city', 'neutral_venue', 'team1', 'team2', 'toss_winner',
       'toss_decision'])



# In[120]:


user_input


# In[121]:


user_input = pd.get_dummies(user_input)


# In[122]:


user_input


# In[123]:


user_input = user_input.reindex(columns = X.columns, fill_value=0)


# In[124]:


user_input.shape


# In[129]:


pred = model_ipl.predict(user_input)


# In[130]:


predict = le.inverse_transform(pred)




# In[136]:


pickle.dump(model_ipl,open('temp.pkl','wb'))
model=pickle.load(open('temp.pkl','rb'))
#print(model_ipl.predict())
print(predict[0])




