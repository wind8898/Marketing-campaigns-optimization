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
from joblib import dump, load
import threading

"""
we have prepared the leads and opportunity data sets 
and we combine the data sets to form machine learning data set
"""


def combinedata():
    print("generating the DataFrame")

    leads = pd.read_csv('data/leads.csv', low_memory=False)

    opps = pd.read_csv('data/opps.csv', low_memory=False)

    opps['Opportunity'].fillna(value=1, inplace=True)

    leads['Opportunity'].fillna(value=0, inplace=True)

    data = pd.concat([opps, leads])

    data.drop_duplicates(subset=['Id'])

    data.to_csv('data/machine_learning_data.csv')

    return None


# clean the DataFrame
def cleandata():

    print("Preparing the data set...")

    data = pd.read_csv('data/machine_learning_data.csv', low_memory=False)

    # we are only taking 100, 000 sample
    data_opp = data.loc[data['Opportunity'] == 1].sample(n=50000)
    data_leads = data.loc[data['Opportunity'] == 0].sample(n=50000)
    data = pd.concat([data_opp, data_leads])

    # convert birth date to age
    time_value = pd.to_datetime(data['Birth Date'], format='%Y-%m-%d')
    time_value = pd.DatetimeIndex(time_value)
    data['Age'] = 2020 - time_value.year
    data['Age'].fillna(value=data['Age'].mean(), inplace=True)

    # Convert the SFDC Campaigns to "yes" : 1 or "no" : 0
    data['SFDC Campaigns'].fillna(value=0, inplace=True)
    data['SFDC Campaigns'] = np.where(data['SFDC Campaigns'] == 0, 0, 1)

    # Convert 'City of Event' to "yes" : 1 or "no" : 0
    data['City of Event'].fillna(value=0, inplace=True)
    data['Attended Event'] = np.where(data['City of Event'] == 0, 0, 1)

    """
    We could try different combination of features, and we will use the following one to save the calculation time.
    Available variables including [ 'Job Title', 'Company Name', 'Person Status',
           'Person Score', 'Person Source', 'Updated At', 'SFDC Type',
           'Appointment Booked', 'Appointment Showed', 'Application Status',
           'Behavior Score', 'Birth Date', 'Citizenship Status', 'City',
           'City of Event', 'Contact Status', 'Date of Birth', 'Source',
           'Media Group', 'Media SubGroup', 'Opportunity', 'Address Country',
           'Interview Booked', 'Interview Showed', 'Primary Program',
           'SFDC Type.1', 'Unsubscribed', 'Application Fee Status',
           'Application Submitted On', 'Education Agent Name', 'Engagement',
           'Interview held#', 'Institution', 'Application Fee Waived',
           'SFDC Campaigns', 'Age'...
    """
    df = data.loc[:, ('Media Group', 'Primary Program', 'Opportunity', 'Unsubscribed', 'Application Fee Waived', 'Age',
                      'Attended Event')]
    df.fillna(value=0, inplace=True)
    df.dropna(how='any', inplace=True)

    print('The data set is ready...')

    return df

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

    dump(dict, 'DictVectorizer.joblib')

    return x_train, x_test


# Prepare the DataFrame for machine learning
# Data prepossessing
def train_test(df):

    print("Split the train and test data")
    print("-" * 100)

    y = df['Opportunity']

    x = df.drop(axis=1, columns=['Opportunity'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    return x_train, x_test, y_train, y_test


# use knn model
def knn(x_train, x_test, y_train, y_test):
    print("Working on the knn model, it may take a couple minutes to finish the process...")

    # x_train, x_test, y_train, y_test = train_test(df)

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

    print(f'The best score for knn model is {gcscore} and the best parameter is {parameter}')

    print("-" * 100)

    dump(gc, 'knn_model.joblib')

    return None


# Using decision tree

def dec(x_train, x_test, y_train, y_test):
    print("Working on the decision tree model, it may take a couple minutes to finish the process...")

    # x_train, x_test, y_train, y_test = train_test(df)

    dec = DecisionTreeClassifier()

    # x_train, x_test = stand(x_train, x_test)
    # x_train, x_test = mm(x_train, x_test)

    x_train, x_test = dict(x_train, x_test)

    dec.fit(x_train, y_train)

    score = dec.score(x_test, y_test)

    export_graphviz(dec, out_file='tree.dot')

    print(f'The score for dec model is {score}')
    print("-" * 100)

    dump(dec, 'decision_tree_model.joblib')

    return None


# Using Random Forest

def rf(x_train, x_test, y_train, y_test):
    print("Working on the Random Forest model, it may take a couple minutes to finish the process...")

    # x_train, x_test, y_train, y_test = train_test(df)

    # x_train, x_test = stand(x_train, x_test)
    # x_train, x_test = mm(x_train, x_test)

    x_train, x_test = dict(x_train, x_test)

    rf = RandomForestClassifier()

    rf.fit(x_train, y_train)

    score = rf.score(x_test, y_test)

    # using GridSearchCV to eval the result
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], 'max_depth': [5, 8, 15, 25, 30]}

    GC = GridSearchCV(rf, param_grid=param, cv=2)

    GC.fit(x_train, y_train)

    GCscore = GC.score(x_test, y_test)

    parameter = GC.best_params_

    print(f'The best score Random Forest model is {GCscore} and the best parameter is {parameter}')
    print("-" * 100)

    dump(GC, 'student_rf_best_model.joblib')

    return None


# save & load model function
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def main():

    df = cleandata()
    x_train, x_test, y_train, y_test = train_test(df)
    # knn(df)
    # dec(df)
    # rf(df)

    model_knn = threading.Thread(target=knn, args=(x_train, x_test, y_train, y_test,))
    model_dec = threading.Thread(target=dec, args=(x_train, x_test, y_train, y_test,))
    model_rf = threading.Thread(target=rf, args=(x_train, x_test, y_train, y_test,))
    model_knn.start()
    model_dec.start()
    model_rf.start()


if __name__ == "__main__":
    main()

