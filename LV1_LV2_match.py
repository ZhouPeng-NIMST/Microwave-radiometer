
# coding: utf-8

import pandas as pd
import numpy as np
import re
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' %x)


pathBt = r'F:\data\beijing2018\LV\LV1\00_12\2018_00_12.csv'
pathSounding = r'F:\data\beijing2018\探空\插值后\2018_54511(插值后).xlsx'
pathOut = r'F:\data\beijing2018\LV\train\samples.csv'

dataBt = pd.read_csv(pathBt,header=None).T
dataSounding = pd.read_excel(pathSounding,header=None).T

def transform(date):
    date = re.sub(r'/',r'-',date) 
    s = re.search(r' .*$',date)
    if s.group() == ' 08:00':
        date = re.sub(r' .*$','-00',date)
    elif s.group() == ' 20:00':
        date = re.sub(r' .*$','-12',date)
    else:
        print('None')
    return date

columns = []
samples = []
dates = dataBt[0].values
bts = dataBt.iloc[:,1:].values.astype('float')
soundings = dataSounding.iloc[:,1:].values
soundingDate = dataSounding.iloc[:,0].tolist()
for i,date in enumerate(dates):
    s = transform(date)
    if s in soundingDate:
        idx = soundingDate.index(s)
        sample = np.r_[bts[i,:],soundings[idx + 2,:],soundings[idx + 3,:]]
        samples.append(sample)
        columns.append(s)
    else:
        print(s,date)
samples = np.array(samples).astype('float')
samples = samples.round(2)

pd.DataFrame(samples,index= columns).to_csv(pathOut,header = None)

