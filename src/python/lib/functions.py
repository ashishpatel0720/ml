from lib import *

def load_data():
    raw_train = pd.read_csv('../data/train.csv')
    raw_test = pd.read_csv('../data/test.csv')
    return raw_train, raw_test


# this function takes dataframe  and  class to be predicted, and plots all columns 
def graph_all_columns(data, class_name):
    cols=data.columns
    for col in cols:
        distinct_values = len(data[col].unique())
        print("Distinct Values for ",col, ' are :',distinct_values)
        if distinct_values < 10:
            function=sns.barplot
        else:
            function=sns.lineplot
        function(x= col , y=class_name, data=data)
        plt.show()
    