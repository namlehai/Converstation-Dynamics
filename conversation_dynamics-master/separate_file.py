# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:47:41 2018

@author: niccolop
"""
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

dfprolife = pd.read_pickle('data/results_prolife.pkl')
dfprochoice = pd.read_pickle('data/results_prochoice.pkl')


print(dfprolife.shape)
print(dfprolife.columns)

prol = [ txt.split(".") for txt in dfprolife['body'].values]
prol = [item for sublist in prol for item in sublist]          
print("number of sentences in profile dataset: ", len(prol))
with open("prolife.txt", 'w', encoding='utf8') as f:
    for sentence in prol:
        f.write(sentence + '\n')

proc = [ txt.split(".") for txt in dfprochoice['body'].values]
proc = [item for sublist in proc for item in sublist]          
print("number of sentences in prochoice dataset: ", len(proc))
with open("prochoice.txt", 'w', encoding='utf8') as f:
    for sentence in proc:
        f.write(sentence + '\n')



#print("three sentences in pro-life")
#print(prol[0:3])

#import pickle
#with open('senteces.pkl', 'wb') as f:
#    pickle.dump([prol,proc], f)

