
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import re


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


pathIn = r'F:\data\beijing2018\LV\LV2\54511'

filenames = os.listdir(path= pathIn)


# In[ ]:


data_lv2 = []
for filename in filenames:
#     筛选出2018-07月
    if re.search(r'2018-07',filename):
        data = pd.read_csv(os.path.join(pathIn,filename),index_col= False)
#         遍历文件
        for i,o in enumerate(data.values):
            s = transform(o[1])
#             只提取温度和相对湿度
            if s[-2:] in ['00','12'] and o[2] == 11 and data.values[i + 2,2] == 13:
                data_lv2.append([s] + o[11:].tolist() + data.values[i + 2,11:].tolist())
            else:
                continue
    else:
        continue


# In[ ]:


Data_lv2 = pd.DataFrame(data_lv2)
Data_lv2.to_csv('lv2_201807.csv',header= None,index=None)

