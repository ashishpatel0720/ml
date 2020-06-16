from lib import *

def load_data():
    raw_train = pd.read_csv('../data/train.csv')
    raw_test = pd.read_csv('../data/test.csv')
    return raw_train, raw_test


# this function takes dataframe  and  class to be predicted, and plots all columns
def graph_all_columns(train_data, class_name):
    cols=train_data.columns
    print("All columns",cols)
    for col in cols:
        if col is class_name: continue #ommiting class itself
        distinct_values = len(train_data[col].unique())
        print("Distinct Values for ",col, ' are :',distinct_values)
        if distinct_values < 10:
            function=sns.barplot
        else:
            function=sns.lineplot
        function(x= col, y=class_name, data=train_data)
        plt.show()

def showColInfo(df, colName):
    print("Null Values : ", df[colName].isnull().sum())
    print("Value Counts:\n",df[colName].value_counts())

def showDFInfo(df):
    plt.figure(figsize=(20,10))
    # this shows null values in each column of a dataframe through a heatmap ( black tick means null values)
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
