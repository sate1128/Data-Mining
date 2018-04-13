# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:17:06 2018

@author: yachiyang
"""


import pandas as pd
import matplotlib.pyplot as plt

import math
from sklearn.svm import LinearSVC
import numpy as np
#os.chdir("C:\\Users\\yachi\\Desktop\\f17_1\\Data Mining\\final")
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
db=pd.DataFrame(pd.read_csv('Preprocessed_character_data.csv'))
#db=pd.DataFrame(pd.read_csv('Preprocessed_character_data_non_alive.csv'))
db.columns

target=db['isAlive']
db=db.drop('isAlive',1)
db=db.drop('name',1)
db=db.drop('isPopular',1)
db=db.drop('boolDeadRelations',1)
db=db.drop('numDeadRelations',1)
#db=db.drop('weight',1)
#db=db.drop('')
train_sample=math.floor(len(db)*0.9)


X_train, X_test, y_train, y_test = train_test_split(db, target, test_size=0.1)

"""
TP (true positives, i.e. correctly predicted dead characters),
FP (false positives, i.e. alive characters predicted to be dead), 
FN (false negatives, i.e. dead characters predicted to be alive), 
TN (true negatives, i.e. correctly predicted alive characters).
"""
def result(y_predict,ans):   
    ans=np.asarray(y_test)
    TP=FP=FN=TN=0
    for i in range(len(y_predict)):
        if (y_predict[i]==0 and ans[i]==0):
            TN=TN+1
        elif (y_predict[i]==1 and ans[i]==0):
            FN=FN+1
        elif(y_predict[i]==0 and ans[i]==1):
            FP=FP+1
        else:
            TP=TP+1
    Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    F=2*(Precision*Recall)/(Precision+Recall)
    print("Precision: %.5f \n" %Precision)
    print("Recall: %.5f \n" %Recall)
    print("F-Measure: %.5f \n" %F)
    
    

"""
LinearSVC 
"""
lin_clf = LinearSVC()
lin_clf.fit(X_train, y_train)
y_predict = lin_clf.predict(X_test)

result(y_predict,y_test)

print("LinearSVC Score: %.5f" %lin_clf.score(X_test,y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predict))


"""
KNeighborsClassifier  
"""
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_predict=knn.predict(X_test)

result(y_predict,y_test)
print("KNeighborsClassifier Score: %.5f" %knn.score(X_test,y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predict))


"""
DecisionTreeClassifier
"""
DecisionTree=DecisionTreeClassifier()
DecisionTree.fit(X_train, y_train)
y_predict=DecisionTree.predict(X_test)
result(y_predict,y_test)


print("DecisionTreeClassifier Score: %.5f" %DecisionTree.score(X_test,y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predict))



"""
GaussianNB
"""
clf_pf = GaussianNB()
clf_pf.fit(X_train, y_train)
y_predict=clf_pf.predict(X_test)
result(y_predict,y_test)

print("GaussianNB Score: %.5f" %clf_pf.score(X_test,y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predict))


"""
Regression
"""

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predict))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_predict))


"""
Daenerys Targaryen
"""

db2=pd.DataFrame(pd.read_csv('Preprocessed_character_data.csv'))
cht=db2.loc[(db2.name=="Daenerys Targaryen")]
cht=cht.drop('name',1)
cht=cht.drop('isAlive',1)
cht=cht.drop('isPopular',1)
cht=cht.drop('boolDeadRelations',1)
cht=cht.drop('numDeadRelations',1)
cht_predict=knn.predict(cht)


"""
Jon Snow
"""

cht=db2.loc[(db2.name=="Jon Snow")]
cht=cht.drop('name',1)
cht=cht.drop('isAlive',1)
cht=cht.drop('isPopular',1)
cht=cht.drop('boolDeadRelations',1)
cht=cht.drop('numDeadRelations',1)
cht_predict=knn.predict(cht)

