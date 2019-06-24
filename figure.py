
# coding: utf-8

# In[ ]:


import tensorflow as tf
import pandas as pd
import os
from matplotlib import pyplot as plt


# In[ ]:


def get_valloss(model,data,idx_1,idx_2,flag = 0):
    a = data
    b = model.predict(a[:,idx_1])
    c = a[:,idx_2]
    if flag:
        return ((b - c)**2).mean(axis = 0)**0.5
    else:
        return ((b - c)**2).mean()**0.5


# In[ ]:


def get_meanerr(model,data,idx_1,idx_2,flag = 0):
    a = data
    b = model.predict(a[:,idx_1])
    c = a[:,idx_2]
    if flag:
        return (b - c).mean(axis = 0)
    else:
        return (b - c).mean()


# In[ ]:


# 加载模型
model_T = tf.keras.models.load_model('model/model_save_T.h5')
model_H = tf.keras.models.load_model('model/model_save_H.h5')
# 加载数据
path = '54511'
data_201807 = pd.read_csv(os.path.join(path,'lv2_soundings_201807.csv'),header= None)


# In[ ]:


# 提取数据时忽略日期信息、晴雨flag
idx_bt = [1,2,3] + list(range(5,19))
idx_t = list(range(19,112))
idx_h = list(range(112,205))
idx_t_lv2 = list(range(205,298))
idx_h_lv2 = list(range(298,391))


# In[ ]:


# 计算rmse
rmse_T = get_valloss(model_T,data_201807.values,idx_bt,idx_t,flag= 1)
rmse_H = get_valloss(model_H,data_201807.values,idx_bt,idx_h,flag= 1)
mean_T = get_meanerr(model_T,data_201807.values,idx_bt,idx_t,flag= 1)
mean_H = get_meanerr(model_H,data_201807.values,idx_bt,idx_h,flag= 1)
mean_T_lv2 = (data_201807.values[:,idx_t_lv2] - data_201807.values[:,idx_t]).mean(axis = 0)
mean_H_lv2 = (data_201807.values[:,idx_h_lv2] - data_201807.values[:,idx_h]).mean(axis = 0)
rmse_T_lv2 = ((data_201807.values[:,idx_t_lv2] - data_201807.values[:,idx_t])**2).mean(axis = 0)**0.5
rmse_H_lv2 = ((data_201807.values[:,idx_h_lv2] - data_201807.values[:,idx_h])**2).mean(axis = 0)**0.5


# In[ ]:


# height
height = '0.000km 0.010km 0.025km 0.050km 0.075km 0.100km 0.130km 0.160km 0.190km 0.220km 0.250km 0.280km 0.310km 0.340km 0.370km 0.400km 0.430km 0.460km 0.490km 0.520km 0.560km 0.600km 0.640km 0.680km 0.720km 0.760km 0.800km 0.840km 0.880km 0.920km 0.960km 1.000km 1.040km 1.080km 1.120km 1.160km 1.200km 1.260km 1.320km 1.380km 1.440km 1.500km 1.560km 1.620km 1.680km 1.740km 1.800km 1.890km 1.980km 2.170km 2.260km 2.350km 2.430km 2.500km 2.600km 2.700km 2.800km 2.900km 3.000km 3.100km 3.200km 3.300km 3.400km 3.500km 3.650km 3.800km 3.950km 4.100km 4.250km 4.400km 4.550km 4.600km 4.800km 5.000km 5.200km 5.400km 5.600km 5.800km 6.000km 6.300km 6.600km 6.900km 7.200km 7.500km 7.800km 8.100km 8.400km 8.700km 9.000km 9.300km 9.600km 9.800km 10.000km'
height = height.split(' ')
height = [int(float(o[:-2])*1000) for o in height]


# In[ ]:


fig = plt.figure(1,(16,7),100)

font_1 = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 15}

ax_1 = plt.subplot(1,2,1)
ax_1.plot(mean_T,height,'b-*',linewidth= 2)
ax_1.plot(mean_T_lv2,height,'r-*',linewidth= 2)
ax_1.plot(rmse_T,height,'b-o',linewidth= 2)
ax_1.plot(rmse_T_lv2,height,'r-o',linewidth= 2)
ax_1.set_xlabel('error/K',font_1)
ax_1.set_ylabel('height/m',font_1)
ax_1.set_title('Error of Temperature',fontdict= font_1,pad = 10)
# ax_1.legend(handles = ax_1.lines,labels = ['After', 'Before'], loc = 'best',prop = {'family': 'Times New Roman','size': 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax_2 = plt.subplot(1,2,2)
ax_2.plot(mean_T,height,'b-*',linewidth= 2)
ax_2.plot(mean_T_lv2,height,'r-*',linewidth= 2)
ax_2.plot(rmse_H,height,'b-o',linewidth= 2)
ax_2.plot(rmse_H_lv2,height,'r-o',linewidth= 2)
ax_2.set_xlabel('error/%',font_1)
ax_2.set_ylabel('height/m',font_1)
ax_2.set_title('Error of Humidity',fontdict= font_1,pad = 10)
ax_2.legend(handles = ax_2.lines,labels = ['ME-After', 'ME-Before','RMSE-After', 'RMSE-Before'], 
            loc = 'best',bbox_to_anchor = (0.25, 1.20),
            prop = {'family': 'Times New Roman','size': 15},
            ncol=2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

fig.savefig(os.path.join(path,'figures/model_lv2_T_H2.jpg'))
plt.show()

