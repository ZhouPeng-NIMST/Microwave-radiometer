
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


data_lv2 = pd.read_csv(r'lv2_201807.csv',header= None)
data_soundings = pd.read_csv(r'samples_201807.csv',header= None)


# In[ ]:


data_matched = []
# 设置有用数据的索引，剔除地面传感器数据
idx_sounding = [0] + list(range(19,205))
for o in data_soundings.values:
    if o[0] in data_lv2.values[:,0]:
#         索引所在行
        idx_lv2 = np.where(data_lv2.values[:,0] == o[0])[0][0]
        data_matched.append(o[idx_sounding].tolist() + data_lv2.values[idx_lv2,1:].tolist())
    else:
        print('{} no matched'.format(o[0]))
data_matched = np.array(data_matched)


# In[ ]:


pd.DataFrame(data_matched).to_csv('lv2_soundings_201808.csv',header = None,index = None)

