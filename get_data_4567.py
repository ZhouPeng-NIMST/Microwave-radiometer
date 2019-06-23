
# coding: utf-8

# In[ ]:


# 提取2018_456月份的亮温值
# 提取2018——7月份的廓线

import pandas as pd
import os
import re


# In[ ]:


path_bt_456 = r'54399/samples.csv'
path_bt_7 = r'54399/samples.csv'
path_profile = r'after_QC\54399\InverseLV2csv'
mon_list_456 = ['2018-04','2018-05','2018-06']
mon_list_07 = ['2018-07']


# In[ ]:


def getBt_of_chosed(path_bt,mon_list):
    data = pd.read_csv(path_bt,header= None,index_col= None)

    idx = []
    for i,simple in enumerate(data.values):
        if simple[0][:7] in mon_list:
            idx.append(i)

    idx_bt = [0,1,2,3] + list(range(5,19))
    idx_t = list(range(19,112))
    idx_h = list(range(112,205))

    data_out = data.iloc[idx,idx_bt]
    return data_out


# In[ ]:


data_2018456 = getBt_of_chosed(path_bt_456,mon_list_456)
data_2018456.to_csv(r'after_QC/54399/bt_2018456.csv',index= None,header= None)


# In[ ]:


data_20187 = getBt_of_chosed(path_bt_7,mon_list_07)
data_20187.to_csv(r'after_QC/54399/bt_20187.csv',index= None,header= None)


# In[ ]:


def transform(date):
    date = re.sub(r'/',r'-',date) 
    s = re.search(r' .*$',date)
    if s.group() == ' 08:00':
        date = re.sub(r' .*$','-00',date)
        return date
    elif s.group() == ' 20:00':
        date = re.sub(r' .*$','-12',date)
        return date
    else:
        return('None')


# In[ ]:


def getProfile_of_chosed(path_profile):
    data_lv2 = []
    filenames = os.listdir(path_profile)
    for filename in filenames:
        path = os.path.join(path_profile,filename)
        data = pd.read_csv(path,header= None,skiprows= 1)
        for i,o in enumerate(data.values):
            s = transform(o[1])
            if s[-2:] in ['00','12'] and o[2] == 11 and data.values[i + 2,2] == 13:
                data_lv2.append([s] + o[11:-2].tolist() + data.values[i + 2,11:-2].tolist())
            else:
                continue
    return pd.DataFrame(data_lv2,columns= None,index= None)


# In[ ]:


lv2_t_h = getProfile_of_chosed(path_profile)


# In[ ]:


lv2_t_h.to_csv('after_QC/54399/After_QC_54399.csv',index= None,header= None)

