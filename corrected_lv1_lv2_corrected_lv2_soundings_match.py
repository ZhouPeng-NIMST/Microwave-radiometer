
# coding: utf-8

# In[1]:


import pandas as pd


# In[73]:


path_lv1_corrected = r'54511/bt/bt_201807_corrected.csv'
path_lv2 = r'54511/After_QC_54511.csv'
path_soundings = r'2018_456(插值后)_long.xlsx'
path_out = r'201807_lv1_lv2_soundings_matched.csv'


# In[9]:


data_lv1 = pd.read_csv(path_lv1_corrected,index_col= None ,header = None)
data_lv2 = pd.read_csv(path_lv2,index_col= None ,header = None)
data_soundings = pd.read_excel(path_soundings,index_col= None,header= None).T


# In[67]:


def match_lv1_lv2_soundings(data_lv1,data_lv2,data_soundings):
    matched_out = []
    for date in data_lv1.values:
#         print(date[0])
        if date[0] in data_lv2[0].values and date[0] in data_soundings[0].values:
            idx_lv2 = np.where(data_lv2[0] == date[0])[0][0]
            idx_soud_T = np.where(data_soundings[0] == date[0])[0][0]+2
            idx_soud_H = np.where(data_soundings[0] == date[0])[0][0]+3
            matched_out.append(date.tolist() + 
                               data_lv2.iloc[idx_lv2,:].values.tolist() + 
                               data_soundings.iloc[idx_soud_T,1:].values.tolist() + 
                               data_soundings.iloc[idx_soud_H,1:].values.tolist())
            print(idx_lv2,idx_soud_T,idx_soud_H)
        else:
            print('{} no matched'.format(date[0]))
    return pd.DataFrame(matched_out,index= None ,columns= None)


# In[ ]:


lv1_lv2_soundings_matched = match_lv1_lv2_soundings(data_lv1,data_lv2,data_soundings)


# In[74]:


lv1_lv2_soundings_matched.to_csv(path_out,index = None,header = None)

