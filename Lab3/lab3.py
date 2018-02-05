#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:01:14 2018

@author: k1461506
"""
import os 
import pandas as pd

titanic = pd.read_csv('data/titanic.csv', header=0, engine='python')


# Check Head
titanic.head()