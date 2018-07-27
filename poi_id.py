#!/usr/bin/python

import sys
import pickle
import csv
import matplotlib.pyplot
sys.path.append("../tools/")
from numpy import mean

from feature_format import featureFormat, targetFeatureSplit

from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'director_fees'] 

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi']

poi_label = ['poi']

features_list = poi_label + email_features + financial_features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Print csv file with info
def make_csv(data_dict):
    """ generates a csv file from a data set"""
    fieldnames = ['name'] + data_dict.itervalues().next().keys()
    with open('data.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in data_dict:
            person = data_dict[record]
            person['name'] = record
            assert set(person.keys()) == set(fieldnames)
            writer.writerow(person)
            
# make_csv(data_dict)
# How many data points (people)?
print"Data points: %i" % len(data_dict)
# For each person, how many features are available?
print "Features avaliables: %i" % len(data_dict["SKILLING JEFFREY K"])
#print enron_data["SKILLING JEFFREY K"]
print "Numer of POI: %i" % len(dict((key, value) for key, value in data_dict.items() if value["poi"] == True))
#print dict((key, value) for key, value in enron_data.items() if value["poi"] == True)

# Are there features with many missing values? etc.
all_features = data_dict[data_dict.keys()[0]].keys()
print("There are %i features for each person in the dataset, and %i features \
are used" %(len(all_features), len(features_list)))
# Are there features with many missing values? etc.
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
    
for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature] == "NaN":
            missing_values[feature] += 1

print("The number of missing values for each feature: ")

for feature in missing_values:
    print("%s: %i" %(feature, missing_values[feature]))
    

### Task 2: Remove outliers
def plot_outliers(data_set, feature_x, feature_y):
    """
    This function takes a dict, 2 strings, and shows a 2d plot of 2 features
    """
    data = featureFormat(data_set, [feature_x, feature_y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'blue'

        matplotlib.pyplot.scatter( x, y, color=color)
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()
    
# Visualize data to identify outliers
#print(plot_outliers(data_dict, 'total_payments', 'total_stock_value'))
#print(plot_outliers(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
#print(plot_outliers(data_dict, 'salary', 'bonus'))
#print(plot_outliers(data_dict, 'total_payments', 'other'))

data_dict.pop("TOTAL", 0)

# Find persons whose features are all "NaN"
email_nan_dict = {}
for person in data_dict:
    email_nan_dict[person] = 0
    for feature in features_list:
        if data_dict[person][feature] == "NaN":
            email_nan_dict[person] += 1
sorted(email_nan_dict.items(), key=lambda x: x[1])

data_dict.pop("LOCKHART EUGENE E", 0)

data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
submit_dict = {}

def compute_fraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if poi_messages == 'NaN':
        poi_messages = 0
    if all_messages == 'NaN':
        return 0
    return float(poi_messages) / float(all_messages)

### Store to my_dataset for easy export below.
my_dataset = data_dict
for person in my_dataset:
    msg_from_poi = my_dataset[person]['from_poi_to_this_person']
    to_msg = my_dataset[person]['to_messages']
    if msg_from_poi != "NaN" and to_msg != "NaN":
        my_dataset[person]['fraction_from_poi'] = msg_from_poi/float(to_msg)
    else:
        my_dataset[person]['fraction_from_poi'] = 0
    msg_to_poi = my_dataset[person]['from_this_person_to_poi']
    from_msg = my_dataset[person]['from_messages']
    if msg_to_poi != "NaN" and from_msg != "NaN":
        my_dataset[person]['fraction_to_poi'] = msg_to_poi/float(from_msg)
    else:
        my_dataset[person]['fraction_to_poi'] = 0


print(plot_outliers(my_dataset, 'fraction_from_poi', 'fraction_to_poi'))

new_features = ["fraction_to_poi", "fraction_from_poi"]  # add two features

new_features_list = features_list + new_features

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list , sort_keys = True)

labels, features = targetFeatureSplit(data)

#Select the best features: 
#Removes all features whose variance is below 80% 
#from sklearn.feature_selection import VarianceThreshold
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#features = sel.fit_transform(features)

#print features
#Removes all but the k highest scoring features
from sklearn.feature_selection import f_classif
k=7
selector = SelectKBest(f_classif, k=7)
selector.fit_transform(features, labels)
print("Best features:")
scores = zip(new_features_list[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print sorted_scores
optimized_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:k]
print(optimized_features_list)

# Extract from dataset without new features
data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# Extract from dataset with new features
data = featureFormat(my_dataset, optimized_features_list + \
                    ['fraction_to_poi', 'fraction_from_poi'], sort_keys = True)
new_labels, new_features = targetFeatureSplit(data)
new_features = scaler.fit_transform(new_features)

from sklearn.metrics import *
#from sklearn.metrics import recall_score
#from sklearn.metrics import accuracy_score

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
def evaluate_clf(grid_search, features, labels, params, iters=100):
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)] 
        pre = pre + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]

    print "accuracy: {}".format(mean(acc))
    print "precision: {}".format(mean(pre))
    print "recall:    {}".format(mean(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))
        
# Naive bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
param = {}
grid_search = GridSearchCV(clf, param)

print("\nNaive bayes model: ")
evaluate_clf(grid_search, features, labels, param)

print("\nNaive bayes model(with new Features): ")
evaluate_clf(grid_search, new_features, new_labels, param)


# SVM
#from sklearn.svm import SVC

#clf = SVC()
#param = {'kernel': ['rbf', 'linear', 'poly'], 'C': [0.1, 1, 10, 100, 1000],\
#           'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'random_state': [42]}
#    
#grid_search = GridSearchCV(clf, param)
#
#print("\nSVM model: ")
#evaluate_clf(grid_search, features, labels, param)

#print("\nSVM model(with new Features): ")
#evaluate_clf(grid_search, new_features, new_labels, param)

# Regression
#from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline

#clf = Pipeline(steps=[
#        ('scaler', preprocessing.StandardScaler()),
#        ('classifier', LogisticRegression())])
#
#param = {'classifier__tol': [1, 0.1, 0.01, 0.001, 0.0001], 
#         'classifier__C': [0.1, 0.01, 0.001, 0.0001]}
#
#grid_search = GridSearchCV(clf, param)

#print("\nRegression model: ")
#evaluate_clf(grid_search, features, labels, param)

#print("\nRegression model(with new Features): ")
#evaluate_clf(grid_search, new_features, new_labels, param)

# K Mean
#from sklearn.cluster import KMeans
#clf = KMeans()
#
#param = {'n_clusters': [1, 2, 3, 4, 5], 'tol': [1, 0.1, 0.01, 0.001, 0.0001],
#         'random_state': [42]}
#
#grid_search = GridSearchCV(clf, param)
#print("\nK-mean model: ")
#evaluate_clf(grid_search, features, labels, param)

#print("\nK-mean model(with new Features): ")
#evaluate_clf(grid_search, new_features, new_labels, param)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn import cross_validation  
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

# Fit data with sklearn decision trees algorithm
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(features_train, labels_train)

# Get the accuracy
#prediction = clf.predict(features_test)
#print accuracy_score(prediction, labels_test)

# Exporting my_classifier.pkl, my_dataset.pkl and my_feature_list.pkl
from tester import dump_classifier_and_data
dump_classifier_and_data(clf, my_dataset, optimized_features_list)
