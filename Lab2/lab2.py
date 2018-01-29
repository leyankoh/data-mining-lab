#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:48:41 2018

@author: k1461506
"""
import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import math


def compute_score(M, X, w, y):
    """
    compute_score() Computes sum of squares error for the model
    input: 
    M = number of instances
    X = list of variable values for M instances
    w = list of parameter values
    y = list of target values
    output: Score (Scalar)
    """
    score = 0 #initialise with empty score
    for j in range(0, M):
        yhat = w[0] + (w[1] * X[j])
        score = score + ((y[j] - yhat)**2 )
        
    score = score/2.0
    return score


def gradient_descent_2(M, X, w, y, alpha):
    """
    Solves linear regression with gradient descent for 2 parameters.
    Input:
    M = number of instances
    X = list of variable values for M instances
    w = list of parameter values (of size 2)
    y = list of target values
    alpha = learning rate
    Output:
    w = updated list of parameter values
    """

    for j in range(0, M):
        yhat = w[0] + (w[1] * X[j])
        error = y[j] - yhat
        w[0] = w[0] + (alpha * error * (1.0/M))
        w[1] = w[1] + (alpha * error * X[j] * (1.0/M))
        
    return w 

# Compute score of regression model 
def r_squared(M, X, w, y): 
   """
   M = number of instances
   X = list of variable values for M instances
   w = list of parameter values
   y = list of target values
   """
   u = compute_score(M, X, w, y) * 2
   
   v = 0
   for j in range(0, M): 
       ymean = np.mean(y)
       v = v + ((y[j] - ymean) ** 2)
       
   r2 = 1 - (u/v)

   return r2

    
    
os.chdir("/home/k1461506/Data Mining Lab/Lab2/")


london = pd.read_csv(os.path.join("practical2-data", "london-borough-profiles-jan2018.csv"))

london.drop(0, inplace=True)
london.replace('.', np.nan, inplace=True)
london.reset_index(inplace=True)
london.drop(london.columns[0], axis=1, inplace=True)

#plot women vs men age 
plt.scatter(london.iloc[:,[70]].astype(float), london.iloc[:,[71]].astype(float))
plt.xlabel("age(men)")
plt.ylabel("age(women)")
plt.title("raw data")
plt.show()


n = 100
# set values to feed into function
X = [float(i) for i in london[london.columns[70]].tolist() if not math.isnan(float(i))]
M = len(X)
w = [0,0]
y = [float(i) for i in london[london.columns[71]].tolist() if not math.isnan(float(i))]
alpha = 0.0001
while n > 0: 
    w = gradient_descent_2(M, X, w, y, alpha)
    n -= 1

print w 

yhats = []
for value in X:
    yhat = w[0] + (w[1] * value) 
    yhats.append(yhat)
    

plt.scatter(london.iloc[:,[70]].astype(float), london.iloc[:,[71]].astype(float))
plt.plot(X, yhats, color='r', linestyle='-.')
plt.xlabel("age(men)")
plt.ylabel("age(women)")
plt.title("raw data")
plt.show()


print "R2 score: "
print r_squared(M, X, w, y)

### Testing Over to see if R2 increases ###
n = 100
# set values to feed into function

X = [float(i) for i in london[london.columns[70]].tolist() if not math.isnan(float(i))]
M = len(X)
w = [0,0]
y = [float(i) for i in london[london.columns[71]].tolist() if not math.isnan(float(i))]
alpha = 0.0001

r2s = []

while n > 0: 
    w = gradient_descent_2(M, X, w, y, alpha)
    r2s.append(r_squared(M, X, w, y))
    n -= 1   

plt.plot(range(0,100), r2s)
plt.xlabel('Number of iterations')
plt.ylabel('R-squared value')


X = [float(i) for i in london[london.columns[70]].tolist() if not math.isnan(float(i))]
M = len(X)
w = [0,0]
y = [float(i) for i in london[london.columns[71]].tolist() if not math.isnan(float(i))]
alpha = 0.001

gradient_descent_2(M, X, w, y, alpha)
