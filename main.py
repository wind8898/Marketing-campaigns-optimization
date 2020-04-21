import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier


# Obtain and combine database
def combinedata():
    print("generating the DataFrame")

    leads = pd.read_csv('data/leads.csv', low_memory=False)

    opps = pd.read_csv('data/opps.csv', low_memory=False)

    opps['Opportunity'].fillna(value=1, inplace=True)

    student = pd.concat([opps, leads])

    student.drop_duplicates(subset=['Id'])

    data = student.iloc[:10000]

    return data


# define the DataFrame for machine learning
data = combinedata()


# clean the DataFrame
def cleandata():
    print("working on prepossessing of DataFrame")

    # Convert the SFDC Campaigns to yes or no
    data['SFDC Campaigns'].fillna(value=0, inplace=True)
    data['From SFDC Campaigns'] = np.where(data['SFDC Campaigns'] == 0, 0, 1)

    # Convert 'City of Event' to yes or no
    data['City of Event'].fillna(value=0, inplace=True)
    data['Attended Event'] = np.where(data['City of Event'] == 0, 0, 1)

    # convert birth date to age
    time_value = pd.to_datetime(data['Birth Date'], format='%Y-%m-%d')
    time_value = pd.DatetimeIndex(time_value)
    data['Age'] = 2020 - time_value.year
    data['Age'].fillna(value=data['Age'].mean(), inplace=True)

    # clean all the features we need for machine learning
    data['Unsubscribed'].fillna(value=0, inplace=True)
    data['Person Score'].fillna(value=0, inplace=True)
    data['Behavior Score'].fillna(value=0, inplace=True)
    data['Media SubGroup'].fillna(value=0, inplace=True)
    data['Address Country'].fillna(value=0, inplace=True)
    data['Primary Program'].fillna(value=0, inplace=True)
    data['Engagement'].fillna(value=0, inplace=True)
    data['Opportunity'].fillna(value=0, inplace=True)
    return None


# set up data prepossessing functions
def im(x):
    im = SimpleImputer(missing_values=np.nan, strategy='mean')

    array = im.fit_transform(x)

    print(array)

    return array


def pca(x):
    pca = PCA(n_components=0.9)

    array = pca.fit_transform(x)

    print(array)

    return array


def var(x):
    var = VarianceThreshold(threshold=0.0)

    array = var.fit_transform(x)

    print(array)

    return None


def mm(x_train, x_test):
    mm = MinMaxScaler()

    x_train = mm.fit_transform(x_train)

    x_test = mm.transform(x_test)

    return x_train, x_test


def stand(x_train, x_test):
    std = StandardScaler()

    x_train = std.fit_transform(x_train)

    x_test = std.transform(x_test)

    return x_train, x_test


def dict(x_train, x_test):
    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient='records'))

    x_test = dict.transform(x_test.to_dict(orient='records'))

    return x_train, x_test


# Prepare the DataFrame for machine learning
# Data prepossessing
def train_test():
    print("preparing train and test data")

    # we could try different combination of features, and we will use the following one to save the calculation time
    df = data[['Media SubGroup', 'Primary Program', 'Unsubscribed', 'Attended Event', 'Opportunity']]

    y = df['Opportunity']

    x = df.drop(axis=1, columns=['Opportunity'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    return x_train, x_test, y_train, y_test


# use knn model
def knn():
    print("working on the knn model")

    x_train, x_test, y_train, y_test = train_test()

    knn = KNeighborsClassifier()

    # x_train, x_test = stand(x_train, x_test)
    # x_train, x_test = mm(x_train, x_test)

    x_train, x_test = dict(x_train, x_test)

    knn.fit(x_train, y_train)

    score = knn.score(x_test, y_test)

    # using GridSearchCV 
    param = {'n_neighbors': [5, 10, 50, 100, 500]}

    gc = GridSearchCV(knn, param_grid=param, cv=2)

    gc.fit(x_train, y_train)

    gcscore = gc.score(x_test, y_test)

    parameter = gc.best_params_

    # y_predict = knn.predict(x_test)
    # y_predict

    print(f'the best score for knn model is {gcscore} and the best parameter is {parameter}')
    print("-" * 100)

    return None


# Using decision tree

def dec():
    print("working on the decision tree model, it may take a couple minutes to finish the process")

    x_train, x_test, y_train, y_test = train_test()

    dec = DecisionTreeClassifier()

    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient='records'))

    x_test = dict.transform(x_test.to_dict(orient='records'))

    dec.fit(x_train, y_train)

    score = dec.score(x_test, y_test)

    export_graphviz(dec, out_file='tree.dot')

    print(f'the score for dec model is {score}')
    print("-" * 100)

    return None


# Using Random Forest

def rf():
    print("working on the Random Forest model, it may take a couple minutes to finish the process")

    x_train, x_test, y_train, y_test = train_test()

    rf = RandomForestClassifier()

    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient='records'))

    x_test = dict.transform(x_test.to_dict(orient='records'))

    rf.fit(x_train, y_train)

    score = rf.score(x_test, y_test)

    # using GridSearchCV to eval the result
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], 'max_depth': [5, 8, 15, 25, 30]}

    GC = GridSearchCV(rf, param_grid=param, cv=2)

    GC.fit(x_train, y_train)

    GCscore = GC.score(x_test, y_test)

    parameter = GC.best_params_

    print(f'the best score Random Forest model is {GCscore} and the best parameter is {parameter}')
    print("-" * 100)

    return rf


# save the random forest model
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == "__main__":
    combinedata()
    cleandata()
    train_test()
    knn()
    # dec()
    # rf()
    # storeTree(rf(), "student_rf_save.pkl")
