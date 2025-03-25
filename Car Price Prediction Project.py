#!/usr/bin/env python
# coding: utf-8

# # Car price prediction case study

# By Vishal Kumar
# 
# Linkedin: https://www.linkedin.com/in/vishal-kumar-819585275/

# #### The flow of the case study is as below:
# 
# 1. Reading the data in python
# 2. Defining the problem statement
# 3. Identifying the Target variable
# 4. Looking at the distribution of Target variable
# 5. Basic Data exploration
# 6. Rejecting useless columns
# 7. Visual Exploratory Data Analysis for data distribution (Histogram and Barcharts)
# 8. Feature Selection based on data distribution
# 9. Outlier treatment
# 10. Missing Values treatment
# 11. Visual correlation analysis
# 12. Statistical correlation analysis (Feature Selection)
# 13. Converting data to numeric for ML
# 14. Sampling and K-fold cross validation
# 15. Trying multiple Regression algorithms
# 16. Selecting the best Model

# #### Data description
# The business meaning of each column in the data is as below
# 
# Price: The Price of the car in dollars
# 
# Age: The age of the car in months
# 
# KM: How many KMS did the car was used
# 
# FuelType: Petrol/Diesel/CNG car
# 
# HP: Horse power of the car
# 
# MetColor: Whether car has metallic color or not
# 
# Automatic: Whether car has automatic transmission or not
# 
# CC: The engine size of the car
# 
# Doors: The number of doors in the car
# 
# Weight: The weight of the car

# ## Important Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('CarPricesData.csv')
data


# In[3]:


data.shape


# In[4]:


data.head(10)


# In[5]:


data.info()


# In[6]:


data.describe(include='all')


# In[7]:


data.isna().sum()   #isnull()


# In[8]:


data.nunique()


# In[9]:


data['HP'].unique()


# In[10]:


data


# ## EDA

# In[11]:


data.columns


# In[12]:


sns.distplot(data.HP)


# In[13]:


data.HP.value_counts()


# In[14]:


# def plots(data,colstoplot):
#         fig,subplot=plt.subplot(nrows=1,ncols = len(colstoplot),figsize=(20,6))
#         fig.suptitle('Bar charts of:' +str(colstoplot))
        
#         for i,j in zip(colstoplot,range(len(colstoplot))):
#             data.groupby(i).size().plot(kind='bar',ax=subplot[j])


# In[15]:


# plots(data=data,colstoplot=['FuelType','HP','Metcolor','Automatic','CC','Doors'])


# In[16]:


data.columns


# In[17]:


sns.distplot(data.Price)


# In[18]:


sns.distplot(data.Age)


# In[19]:


data.Weight.hist()


# In[20]:


sns.countplot(data.CC)


# In[21]:


# outliers

data.Weight.describe()


# In[22]:


sns.boxplot(data.Weight)


# In[23]:


data=data[data['Weight']<1150]
sns.boxplot(data.Weight)


# In[24]:


sns.displot(data.HP, kde=True)


# ## missing value treatment

# In[25]:


data.isna().sum()


# In[26]:


data[data.Age.isna()]


# In[27]:


data.Age.fillna(0,inplace=True)


# In[28]:


data.isna().sum()


# In[29]:


data.FuelType.fillna('NA',inplace=True)


# In[30]:


data[data.CC.isna()]


# In[31]:


data['CC']=np.where(data['HP']==110,1600.0,data['CC'])
data['CC']=np.where(data['HP']==86,1300.0,data['CC'])


# In[32]:


sns.heatmap(data.corr(),annot=True)


# In[33]:


data.head()


# In[34]:


data=pd.get_dummies(data)
data.head()


# In[35]:


x=data.drop(columns='Price')
y=data['Price']


# In[36]:


x.head(2)


# ## Train and test split

# In[37]:


from sklearn.preprocessing import MinMaxScaler


# In[38]:


pred = MinMaxScaler()
fit= pred.fit(x)
x= fit.transform(x)
x


# In[39]:


from sklearn.model_selection import train_test_split


# In[40]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# In[41]:


x_train


# In[42]:


x_test


# In[43]:


##Modeling

from sklearn.linear_model import LinearRegression

RegModel = LinearRegression()
fit= RegModel.fit(x_train,y_train)
y_pred=fit.predict(x_test)


# In[44]:


pd.DataFrame(y_pred)


# In[45]:


pd.DataFrame(y_test)


# In[46]:


from sklearn import metrics


# In[47]:


print('R2 score',metrics.r2_score(y_test,y_pred) )


# In[48]:


## Model 2
from sklearn.tree import DecisionTreeRegressor


RegModel = DecisionTreeRegressor()
fit= RegModel.fit(x_train,y_train)
y_pred=fit.predict(x_test)
print('R2 score',metrics.r2_score(y_test,y_pred) )
pd.DataFrame(y_pred)


# In[49]:


pd.DataFrame(y_test)


# In[50]:


### Model3
from sklearn.ensemble import RandomForestRegressor
RegModel = RandomForestRegressor()
fit= RegModel.fit(x_train,y_train)
y_pred=fit.predict(x_test)
print('R2 score',metrics.r2_score(y_test,y_pred) )
pd.DataFrame(y_pred)


# In[51]:


## model4
from sklearn.ensemble import AdaBoostRegressor
DTR = RandomForestRegressor()
model = AdaBoostRegressor(n_estimators=100,base_estimator=DTR, learning_rate =0.04)


fit= model.fit(x_train,y_train)
y_pred=fit.predict(x_test)
print('R2 score',metrics.r2_score(y_test,y_pred) )
pd.DataFrame(y_pred)


# In[52]:


## Model5
from xgboost import XGBRegressor

model = XGBRegressor(max_depth=  5, learning_rate=0.1, n_estimators= 100, objective='reg:linear',booster='gbtree')


fit= model.fit(x_train,y_train)
y_pred=fit.predict(x_test)
print('R2 score',metrics.r2_score(y_test,y_pred) )
pd.DataFrame(y_pred)


# In[53]:


### Model3
from sklearn.ensemble import RandomForestRegressor
final_model = RandomForestRegressor()
fit= final_model.fit(x_train,y_train)
y_pred_final=fit.predict(x_test)

print('R2 score',metrics.r2_score(y_train,fit.predict(x_train)) )
print('R2 score',metrics.r2_score(y_test,y_pred_final) )



##gridsearch cv or randomize search cv


# In[54]:


data.corr()


# In[55]:


data


# In[56]:


## feature engineering
x=data.drop(columns=['Price','CC','FuelType_NA'])
y=data['Price']

x.head(2)


# In[57]:


pred = MinMaxScaler()
fit= pred.fit(x)
x= fit.transform(x)
x

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

x_train

x_test

### Model3
from sklearn.ensemble import RandomForestRegressor
final_model = RandomForestRegressor()
fit= final_model.fit(x_train,y_train)
y_pred_final=fit.predict(x_test)

print('Base model accuracies',0.8744112420928829)

# print('R2 score',metrics.r2_score(y_train,fit.predict(x_train)) )
print('R2 score',metrics.r2_score(y_test,y_pred_final) )


##gridsearch cv or randomize search cv


# In[58]:


#deploy
# 1. flask api
# 2. prediction
# 3. final model


# In[59]:


# save
import joblib
joblib.dump(fit, "car_price_final_model.joblib")


# In[60]:


# load, no need to initialize the loaded_rf
loaded_rf = joblib.load("car_price_final_model.joblib")


# In[61]:


data.info()


# In[62]:


### prediction code
import pandas as pd
Age= float(input('Enter age of your car'))
KM= int(input('Enter number of km'))
FuelType= input('fuel type of your car:-Diesel, Petrol or CNG ')
HP= int(input('Enter value of HP'))
MetColor= int(input('Enter the value for metcolor'))
Automatic= int(input('Enter the value for Automatic'))
Doors= int(input('Enter the value for Doors'))
Weight= float(input('Enter the value for weight'))

# # input_series= pd.Series([Age,KM,HP,MetColor,Automatic,Doors,Weight,FuelType])
# input_indexes = pd.Series['Age','KM','HP','MetColor','Automatic','Doors','Weight','FuelType']

#Creating a dictionary by passing Series objects as values
frame = {'Age':[Age],'KM':[KM],'HP':[HP],'MetColor':[MetColor],'Automatic':[Automatic],'Doors':[Doors],'Weight':[Weight],'FuelType':[FuelType]}
#Creating DataFrame by passing Dictionary
Test_data = pd.DataFrame.from_dict(frame)
# #Printing elements of Dataframe
# print(result)
Test_data

# prediction code

# import pandas as pd

# Age= 25.2
# KM= 45785
# FuelType= 'Diesel'
# HP= 90
# MetColor= 1
# Automatic= 0
# Doors= 3
# Weight= 1165.2


# frame = {'Age':[Age],'KM':[KM],'HP':[HP],'MetColor':[MetColor],'Automatic':[Automatic],'Doors':[Doors],'Weight':[Weight],'FuelType':[FuelType]}
# #Creating DataFrame by passing Dictionary
# Test_data = pd.DataFrame.from_dict(frame)
# # #Printing elements of Dataframe
# # print(result)
# Test_data
# # Prediction

# In[63]:


def predciction_code(Test_data):
    test=pd.get_dummies(Test_data)
    if 'FuelType_Diesel' not in test.columns:
        test['FuelType_Diesel'] = 0
    if 'FuelType_Petrol' not in test.columns:
        test['FuelType_Petrol'] = 0
    if 'FuelType_CNG' not in test.columns:
        test['FuelType_CNG'] = 0

    pred = MinMaxScaler()
    fit= pred.fit(x)
    test= fit.transform(test)
    pred_new=loaded_rf.predict(test)
    return pred_new

predciction_code(Test_data)

