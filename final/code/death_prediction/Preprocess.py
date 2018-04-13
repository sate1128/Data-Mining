# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:43:55 2018

@author: yachiyang
"""

import pandas as pd
import matplotlib.pyplot as plt


import numpy as np
import os
os.chdir("C:\\Users\\yachi\\Desktop\\f17_1\\Data Mining\\final")


db=pd.DataFrame(pd.read_csv('character-predictions_v5.csv'))

db.columns
#第0col是人物編號
"""
丟掉不要的欄位

"""
db.drop(db.columns[[0,1,2,3,4,5,7,10,11,12,13,14,16,28]], axis=1, inplace=True)


"""
處理映射 one-hot encoding
"""

onehot_encoding=pd.get_dummies(db['house'],prefix='house')
db=db.drop('house',1)
onehot_db=pd.concat([db, onehot_encoding],axis=1)



onehot_encoding=pd.get_dummies(onehot_db['culture'],prefix='culture')
onehot_db=onehot_db.drop('culture',1)
onehot_db=pd.concat([onehot_db, onehot_encoding],axis=1)


onehot_db.iloc[0]


onehot_encoding=pd.get_dummies(onehot_db['overlord(AGOT)'],prefix='overlord(AGOT)')
onehot_db=onehot_db.drop('overlord(AGOT)',1)
onehot_db=pd.concat([onehot_db, onehot_encoding],axis=1)



onehot_encoding=pd.get_dummies(onehot_db['overlord(ADWD)'],prefix='overlord(ADWD)')
onehot_db=onehot_db.drop('overlord(ADWD)',1)
onehot_db=pd.concat([onehot_db, onehot_encoding],axis=1)





"""
處理性別的onehot encoding:
    如果是Male, Female, Not Specified因為這三種都是等價的關係因此需要找一個方法讓這三個屬性距離原點是相同距離
    不能使用0和1
"""


onehot_encoding=pd.get_dummies(onehot_db['male'],prefix='Gender')
#onehot_encoding.head()
onehot_db=onehot_db.drop('male',1)
onehot_db

onehot_db=pd.concat([onehot_db, onehot_encoding],axis=1)

onehot_db.iloc[2]


"""
處理資料缺失
"""

def set_mother_alive(df):
    df.loc[(df.isAliveMother.notnull()),"alive_mother_known"]="1"
    df.loc[(df.isAliveMother.isnull()),"alive_mother_known"]="0"
    df=df.drop('isAliveMother',1)
    return df

def set_father_alive(df):
    df.loc[(df.isAliveFather.notnull()),"alive_father_known"]="1"
    df.loc[(df.isAliveFather.isnull()),"alive_father_known"]="0"
    df=df.drop('isAliveFather',1)
    return df

def set_heir_alive(df):
    df.loc[(df.isAliveHeir.notnull()),"alive_heir_known"]="1"
    df.loc[(df.isAliveHeir.isnull()),"alive_heir_known"]="0"
    df=df.drop('isAliveHeir',1)
    return df

def set_spouse_alive(df):
    df.loc[(df.isAliveSpouse.notnull()),"alive_spouse_known"]="1"
    df.loc[(df.isAliveSpouse.isnull()),"alive_spouse_known"]="0"
    df=df.drop('isAliveSpouse',1)
    return df


data=set_mother_alive(onehot_db)
data=set_father_alive(data)
data=set_heir_alive(data)
data=set_spouse_alive(data)

data=onehot_db.drop('isAliveMother',1)
data=data.drop('isAliveFather',1)
data=data.drop('isAliveHeir',1)
data=data.drop('isAliveSpouse',1)

data.head()

#data.to_csv('Preprocessed_character_data_non_alive.csv', index=False)
data.to_csv('Preprocessed_character_data.csv', index=False)

