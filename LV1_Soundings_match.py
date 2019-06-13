
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' %x)


# In[2]:


pathBt = r'F:\data\beijing2018\LV\LV1\00_12\2018_00_12.csv'
pathSounding = r'F:\data\beijing2018\探空\插值后\2018_54511(插值后)_long.xlsx'
pathOut = r'F:\data\beijing2018\LV\train\samples.csv'


# In[3]:


dataBt = pd.read_csv(pathBt,header=None).T
dataSounding = pd.read_excel(pathSounding,header=None).T


# In[4]:


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


# In[5]:


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


# In[6]:


pd.DataFrame(samples,index= columns).to_csv(pathOut,header = None)


# In[ ]:


# 汇总处理
data = pd.read_csv(pathOut,header= None)

# 挑出2018-07的数据，后期要对比，不参与训练
dates = data[0].values
idx_201807 = []
idx_others = []
for i in range(len(dates)):
    if dates[i][:7] == '2018-07':
        idx_201807.append(i)
    else:
        idx_others.append(i)

# 保存
data.iloc[idx_201807].to_csv(r'samples_201807.csv',index = None,header = None)
data.iloc[idx_others].to_csv(r'samples_others.csv',index = None,header = None)

