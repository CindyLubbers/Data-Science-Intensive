#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
from sklearn import svm
from sklearn.metrics import accuracy_score

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

### C = 10
#clf = svm.SVC(kernel='rbf', C=10)
#t0 = time()
#clf.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#
#t0 = time()
#pred = clf.predict(features_test)
#print "prediction time:", round(time()-t0, 3), "s"
#
#accuracy = accuracy_score(labels_test, pred)
#print 'C = 10'
#print accuracy
#
####C = 100
#clf = svm.SVC(kernel='rbf', C=100)
#t0 = time()
#clf.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#
#t0 = time()
#pred = clf.predict(features_test)
#print "prediction time:", round(time()-t0, 3), "s"
#
#accuracy = accuracy_score(labels_test, pred)
#print 'C = 100'
#print accuracy
#
#### C = 1000
#clf = svm.SVC(kernel='rbf', C=1000)
#t0 = time()
#clf.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#
#t0 = time()
#pred = clf.predict(features_test)
#print "prediction time:", round(time()-t0, 3), "s"
#
#accuracy = accuracy_score(labels_test, pred)
#print 'C = 1000'
#print accuracy

### C = 10000
clf = svm.SVC(kernel='rbf', C=10000)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

accuracy = accuracy_score(labels_test, pred)
print 'C = 10000'
print accuracy
print pred[[10, 26, 50]], "are the predictions for the 10th, 26th and 50th row"
print sum(pred), "are the total number of cases predicted in the Chris class"
#########################################################


