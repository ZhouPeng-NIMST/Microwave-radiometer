
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


path_lv1 = r'54399/bt/bt_2018456.csv'
path_mono = r'54399/bt/2018_456(BT_mono).xlsx'
path_out_lv1_mono_matched = r'54399/bt/lv1_mono_matched.csv'
path_out_correct_model = r'54399/bt/path_correct_model'
path_bt_07 = r'54399/bt/bt_20187.csv'


# In[ ]:


data_lv1 = pd.read_csv(path_lv1,header= None,index_col= None)
# 把地面传感器数据传到后面
data_lv1[1] = data_lv1[1].apply(lambda x: x + 273.15)
data_lv1[18] = data_lv1[1]
data_lv1[19] = data_lv1[3]
data_lv1[20] = data_lv1[2]
data_lv1 = data_lv1.drop([1,2,3],axis=1)


# In[ ]:


data_mono = pd.read_excel(path_mono,header= None,index_col= 0).T


# In[ ]:


def match_momo_lv1(data_lv1,data_mono):
    bt_matched_out = []
    for i,lv1 in enumerate(data_lv1):
        if lv1[0] in data_mono[:,0]:
            bt_matched_out.append(lv1.tolist() + data_mono.tolist()[data_mono[:,0].tolist().index(lv1[0])])
    return pd.DataFrame(bt_matched_out)


# In[ ]:


bt_matched_out = match_momo_lv1(data_lv1.values,data_mono.values)

# 删除几个误差比较大的
abs_err = abs(bt_matched_out[7] - bt_matched_out[25])
idx_del = np.where(abs_err >= 0.9*(abs_err.max()))[0]
bt_matched_out = bt_matched_out.drop(idx_del,axis= 0)


# In[ ]:


# 读取待订正的
data_bt_07 = pd.read_csv(path_bt_07,index_col= None,header = None)
data_bt_07[15] = data_bt_07[15].apply(lambda x: x + 273.15)


# In[ ]:


def lv1_correct(pd_arr,lv1_arr):
    fits = []
    df_corrected = lv1_arr.copy()
    for i in range(0,18):
        if i  == 0:
            df_corrected[i] = lv1_arr[i]
            fits.append(0)
        else:
            a = pd_arr[i]
            b = pd_arr[i + 18]
#             print(a,b)
            bias = np.mean(b - a)
#             print(bias)
            z1 = np.polyfit(a+ bias,b,1)
        #     z1 = np.polyfit(bt_matched_out[idx],bt_matched_out[idx + 18],1)
            p1 = np.poly1d(z1)
            # 计算拟合后的值
            c = p1(lv1_arr[i] + bias)
            df_corrected[i] = pd.Series(c.round(2),index=lv1_arr.index)
            fits.append(p1)
    return df_corrected,np.array(fits)


# In[ ]:


# 保存
df_corrected,correct_model = lv1_correct(bt_matched_out,data_bt_07)
df_corrected.to_csv(path_out_lv1_mono_matched,index= None,header= None)
bt_matched_out.to_csv(path_out_lv1_mono_matched,index= None,header= None)
np.save(path_out_correct_model,correct_model)


# In[ ]:


# 画图
fig = plt.figure(1,(24,24),100)
indexList = list(range(0,16)) + [17]
channel = ['22.2Ghz','23.035Ghz','23.835Ghz','25.5Ghz','26.235Ghz','27.5Ghz','30Ghz','51.25Ghz','52.28Ghz','53.85Ghz','54.94Ghz','56.66Ghz','57.29Ghz','58.8Ghz','T','H']
for i in range(1,17):
    idx = indexList[i]
    fig.add_subplot(4,4,i)
    a = bt_matched_out[idx]
    b = bt_matched_out[idx + 18]
    bias = np.mean(b - a)
    z1 = np.polyfit(a+ bias,b,1)
#     z1 = np.polyfit(bt_matched_out[idx],bt_matched_out[idx + 18],1)
    p1 = np.poly1d(z1)
    # 计算拟合后的值
    c = p1(a + bias)
    r = np.corrcoef(c,b)[0,1]
    plt.plot(a + bias, b, 'ko',label='original values',alpha = 0.6)
    plt.plot(a + bias, c, 'b',label='polyfit values',linewidth = 2,alpha = 1)
    plt.plot([np.min([a,b]),np.max([a,b])],[np.min([a,b]),np.max([a,b])],'k--',label='1:1 line',linewidth = 2,alpha = 1)
    plt.xlabel('DATA_LV1',fontsize = 12,family = 'Times New Roman')
    plt.ylabel('DATA_MONO',fontsize = 12,family = 'Times New Roman')
    #指定legend的位置
    plt.legend(loc=4,prop = {'family': 'Times New Roman','size': 15})
    plt.text(np.min([a,b]),np.max([a,b])-10,s = 'y = ' + str(p1).split('\n')[1] + '\n' + 'r² = ' + str(r.round(2))+ '\nN=30',fontsize = 15)
    plt.title('Channel ' + channel[i - 1],fontsize = 15,family = 'Times New Roman')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
# plt.savefig('correct_54511.jpg')
plt.show()


# In[ ]:


data_bt_07

