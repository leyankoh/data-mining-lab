#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:01:14 2018

@author: k1461506
"""
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

titanic = pd.read_csv('data/titanic.csv', header=0, engine='python')
titanic.dropna(axis=0, how='any', inplace=True)

# 1 for female, 0 for male 
titanic['Sex'].replace('female', 1, inplace=True)
titanic['Sex'].replace('male', 0, inplace=True)

# Check Head
print titanic.head()

# describe data
print titanic.describe()

# Pair plots
g = sns.PairGrid(titanic.loc[:,['Survived','Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].dropna(axis=0, how='any'), hue='Survived')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()

# Split them into analysis set and target set 
target = titanic['Survived']
analysis = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# Split dataset 
from sklearn.model_selection import train_test_split
df_analysis_train, df_analysis_test, df_target_train, df_target_test = train_test_split(analysis, target, test_size=0.4, random_state=0)


# Build decision tree on training data 
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy').fit(df_analysis_train, df_target_train) # fit DT model
titanic_score = clf.score(df_analysis_test, df_target_test) # test DT model
predictions = clf.predict(df_analysis_test)

#plot decision tree
import pydotplus
from IPython.display import Image
dot_data = tree.export_graphviz(clf, out_file = None, filled=True, rounded=True, feature_names=(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']),
                                class_names=(['Yes', 'No']),
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

from sklearn.metrics import confusion_matrix
confusion_matrix(df_target_test, predictions)

# run logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model = model.fit(df_analysis_train, df_target_train) # create LG model

titanic_LG_score = model.score(df_analysis_test, df_target_test) # test LG model

# plot ROC curve 
from sklearn.metrics import roc_curve, auc

predicted = model.predict(df_analysis_test) # store list of predicted values with LG model
fpr, tpr, thresholds = roc_curve(df_target_test, predicted)
roc_auc = auc(fpr, tpr)

# Plot it!
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# 