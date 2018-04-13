# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:25:25 2018

@author: yachiyang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\\yachi\\Desktop\\f17_1\\Data Mining\\final")

edge=pd.DataFrame(pd.read_csv('edges.csv'))
edge.head()

weight=edge.groupby('Source')['Weight'].sum()

cht=pd.DataFrame(pd.read_csv('cht_to_name.csv'))
cht['weight'][0:]=weight

cht.to_csv('cht_to_name_weight.csv')


real_cht=pd.DataFrame(pd.read_csv('character-predictions_v4.csv'))
real_cht.head()
real_cht['weight']=1

cht.loc[0]['weight']
for i in range(len(real_cht)):
        for j in range(len(cht)):
            if(real_cht.loc[i].name==cht.loc[j].Character):
                real_cht.loc[i]['weight']=cht.loc[j]['weight']
                break