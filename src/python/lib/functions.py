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
