import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score,GridSearchCV

import streamlit as st
from sklearn import ensemble
import xgboost

import warnings
import random
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

st.write("# Titanic Prediction")
'Files',os.listdir("../data") # this will list files in input directory

@st.cache
def load_data():
    raw_train=pd.read_csv('../data/train.csv')
    raw_test=pd.read_csv('../data/test.csv')
    return raw_train, raw_test

raw_train,raw_test = load_data()

def show_df(df,name):
    if st.checkbox("Show full data for "+ name, key = name):
        'Total Number of Rows and Columns', len(df),len(df.columns)
        st.write(pd.DataFrame(df.isna().sum(), columns=['Null Count']))
        st.write(df)
    else:
        st.write(df.head())

def show_col(df,colname):
        'Value Counts',data[colname].value_counts()
        'Distinct Values',len(data[colname].unique())
        'No. of elements with null values:',data[colname].isnull().sum()


show_df(raw_train, 'TRAIN')
show_df(raw_test, 'TEST')


st.write("> **We are merging data together, so we can process data, after feature engineering we will split.**")

data=pd.concat([raw_train, raw_test], axis=0).reset_index(drop=True) # concatinating test and train removing index
show_df(data,'DATA')

st.write("### Visualization & Data Cleaning")
st.write("> Cabin")


show_col(data,'Cabin')
data["Cabin"]=data['Cabin'].fillna('X') # adding null to X
data["Cabin"]=data['Cabin'].str.get(0) # trimming all values to first character
show_col(data,'Cabin')

sns.barplot(x="Cabin", y="Survived", data=data) # relation between Cabin and survival
st.pyplot()


st.write("> Embarked")
show_col(data,'Embarked')
data['Embarked']=data['Embarked'].fillna('C')
show_col(data,'Embarked')

sns.barplot(x="Embarked", y="Survived", data=data) # relation between Cabin and survival
st.pyplot()


st.write("> Fare")
show_col(data,'Fare')
data['Fare'] = data.Fare.fillna(data.Fare.median())
show_col(data,'Fare')

sns.kdeplot(data.loc[data['Survived'] == 0, 'Fare'], label='Died')
sns.kdeplot(data.loc[data['Survived'] == 1, 'Fare'], label='Survived')
st.pyplot()


st.write("> Name")
show_col(data,'Name')
# Get Title from Name
data["Title"] = data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
data["Title"] = data["Title"].replace(['Mlle','Ms'], 'Miss')
data["Title"] = data["Title"].replace(['Mme'], 'Mrs')
data["Title"] = data["Title"].replace(['Rev', 'Dr', 'Col', 'Major', 'Capt'], 'Officer')
data["Title"] = data["Title"].replace(['the Countess', 'Don', 'Lady', 'Sir', 'Jonkheer', 'Dona'], 'Royalty')
show_col(data,'Title')
sns.barplot(x='Title', y='Survived', data=data)
st.pyplot()


st.write("> Parch & SibSp")
# Create a family size descriptor from SibSp and Parch
data["Fsize"] = data["SibSp"] + data["Parch"] + 1
show_col(data,'Fsize')
sns.barplot(x='Fsize', y='Survived', data=data);st.pyplot()

# Create new feature of family size
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
data['FamilyLabel']=data['Fsize'].apply(Fam_label)
sns.barplot(x="FamilyLabel", y="Survived", data=data); st.pyplot()

st.write("> Pclass")
show_col(data,'Pclass')
sns.barplot(x="Pclass", y="Survived", data=data); st.pyplot()


st.write("> Sex")
show_col(data,'Sex')
sns.barplot(x="Sex", y="Survived", data=data); st.pyplot()



st.write("> Ticket")
show_col(data,'Ticket')
Ticket_Count = dict(data['Ticket'].value_counts())
data['TicketGroup'] = data['Ticket'].apply(lambda x:Ticket_Count[x])
show_col(data,'TicketGroup')
 # group ticket by count of sales
sns.barplot(x="TicketGroup", y="Survived", data=data); st.pyplot()

def Ticket_Label(s): # group all having same survival together
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

data['TicketGroup'] = data['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=data); st.pyplot()


st.write("> Age")
show_col(data,'Age')

st.write("#### we don't have age of some people and its very important attribute, so predicting using other attributes")

age_df = data[['Age', 'Pclass','Sex','Title','Fsize']]

# get_dummies() only works on strings, and convert to one-hot encoding
# here it will only work on String columns ( Sex(male/female) and title)
age_df=pd.get_dummies(age_df) # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
show_df(age_df,'age_df')

known_age = age_df[age_df.Age.notnull()].as_matrix() # Rows with known age
unknown_age = age_df[age_df.Age.isnull()].as_matrix() # Rows with unknown age
y = known_age[:, 0] # age columns
X = known_age[:, 1:] # all other columns
'Known Age, X',X
'Known Age, y',y

#predicting ages for unknown using XGBoost
rfr=xgboost.XGBClassifier()
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1::])
data.loc[(data.Age.isnull()), 'Age' ] = predictedAges #put predicted ages where it was null
show_df(data,'data')

show_col(data,'Age')
sns.kdeplot(data.loc[data['Survived'] == 0, 'Age'], label='Died')
sns.kdeplot(data.loc[data['Survived'] == 1, 'Age'], label='Survived')
st.pyplot()

st.write("### Modelling")

#choosing required columns and converting to 0/1 for strings
data=data[['Survived','Age','Cabin','Embarked','Fare','Pclass','Sex','TicketGroup','FamilyLabel','Title']]
show_df(data,'final')
data=pd.get_dummies(data) # taking in matrix form for using in model

st.write("> Splitting both train and test data")
train=data[:len(raw_train)] # take train data
test=data[len(raw_train):].drop(['Survived'],axis=1) # take test data
X = train.drop(['Survived'],axis=1)
y = train.Survived


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


st.write("### RandomForest")
random_state= st.slider("Random State", min_value=1, max_value=20, value=10)
n_estimators= st.slider("NEstimator", min_value=10, max_value=100, value=25)
max_depth= st.slider("max_depth", min_value=1, max_value=20, value=5)


model = ensemble.RandomForestClassifier(random_state = random_state,
                                        warm_start = True,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        max_features = 'sqrt')
clf =model.fit(X_train,y_train)
"Score:", clf.score(X_test,y_test)

predictions = model.predict(test)
submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"],
                           "Survived": predictions.astype(np.int32)})

outputFile = "../output/output.csv"
submission.to_csv(outputFile, index=False)

'File written:', outputFile

st.write("## END :)")
