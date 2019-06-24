
# coding: utf-8

# In[ ]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
# import re
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
np.set_printoptions(suppress = True)


# In[ ]:


data_bt_mono = pd.read_excel(r'samples/1317(BT_mono).xlsx',index_col=0)
data_soundings = pd.read_excel(r'samples/1317_54511(插值后)_long.xlsx')
data_test = pd.read_csv(r'201807_lv1_lv2_soundings_matched.csv',header=None,index_col= None)


# In[ ]:


def get_valloss(model,x,y,flag = 0):
    pred = model.predict(x)
    if flag:
        return ((pred - y)**2).mean(axis = 0)**0.5
    else:
        return ((pred - y)**2).mean()**0.5
    
def get_mean_err(model,x,y,flag = 0):
    pred = model.predict(x)
    if flag:
        return (pred - y).mean(axis = 0)
    else:
        return (pred - y).mean()
    
def deleteBigErr(model,samples):
    pred = model.predict(samples[:,0:17])
    result = abs(pred - samples[:,17:]).mean(axis = 1)
    idx = np.where(result>3)[0]
    samples = np.delete(samples,idx,axis=0)
    return samples


# In[ ]:


# 索引出对应的温度与湿度
index_t = list(range(2,data_soundings.shape[1],4))
index_h = list(range(3,data_soundings.shape[1],4))

data_soundings_t = data_soundings.values[:,index_t]
data_soundings_h = data_soundings.values[:,index_h]


# In[ ]:


# 合并
samples_t = np.r_[data_bt_mono,data_soundings_t].T
samples_h = np.r_[data_bt_mono,data_soundings_h].T
# 去掉 nan
idx = np.where(np.isnan(samples_t[:,0]))[0][0]
samples_t = np.delete(samples_t,idx,axis= 0)
samples_h = np.delete(samples_h,idx,axis= 0)


# In[ ]:


# 交叉检验
data_train,_ = train_test_split(samples_h,test_size = 0,random_state = 10,shuffle = False)


# In[ ]:


idx_lv2_T = list(range(19,66))
idx_lv2_T_corrected = list(range(66,113))
idx_lv2_H = list(range(113,160))
idx_lv2_H_corrected = list(range(160,207))
idx_soundings_T = list(range(207,254))
idx_soundings_H = list(range(254,301))
idx_lv1 = list(range(1,18))


# In[ ]:


# 建模
inputs = layers.Input(shape=(17,),name = 'inputs')
hidden_1 = layers.Dense(24,activation= 'linear',name = 'hidden_1')(inputs)
dropout_1 = layers.Dropout(0.1,seed= 10,name = 'dropout_1')(hidden_1)
outputs = layers.Dense(47,activation='linear',name = 'outputs')(dropout_1)

model = tf.keras.Model(inputs= inputs,outputs= outputs,name = 'Model_H')
# 模型摘要
model.summary()
# 模型编译
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= 0.001),
              loss= 'mean_squared_error',
              metrics= [])
# 训练记录
history = model.fit(x = data_train[:,0:17],
                    y = data_train[:,17:],
                    batch_size = None,
                    validation_data= (data_test.values[:,idx_lv1],data_test.values[:,idx_soundings_H]),
                    epochs= 5000,
                    verbose= 1,
                    callbacks= [tf.keras.callbacks.EarlyStopping(monitor= 'val_loss',
                                                                 patience= 200,
                                                                 restore_best_weights= True)])


# In[ ]:


model.save(r'model/H.h5')


# In[ ]:


height = '0km,0.1km,0.2km,0.3km,0.4km,0.5km,0.6km,0.7km,0.8km,0.9km,1km,1.25km,1.5km,1.75km,2km,2.25km,2.5km,2.75km,3km,3.25km,3.5km,3.75km,4km,4.25km,4.5km,4.75km,5km,5.25km,5.5km,5.75km,6km,6.25km,6.5km,6.75km,7km,7.25km,7.5km,7.75km,8km,8.25km,8.5km,8.75km,9km,9.25km,9.5km,9.75km,10km'
height = height.split(',')
height = [int(float(o[:-2])*1000) for o in height]


# In[ ]:


# 绘图 T
mean_test = get_mean_err(model,data_test.values[:,idx_lv1],data_test.values[:,idx_soundings_T],flag= 1)
rmse_test = get_valloss(model,data_test.values[:,idx_lv1],data_test.values[:,idx_soundings_T],flag= 1)
mean_lv2 = (data_test.values[:,idx_lv2_T] - 273.15 - data_test.values[:,idx_soundings_T]).mean(axis=0)
rmse_lv2 = ((data_test.values[:,idx_lv2_T] - 273.15 - data_test.values[:,idx_soundings_T])**2).mean(axis=0)**0.5
mean_lv2_corrected = (data_test.values[:,idx_lv2_T_corrected]- 273.15 - data_test.values[:,idx_soundings_T]).mean(axis=0)
rmse_lv2_corrected = ((data_test.values[:,idx_lv2_T_corrected]- 273.15 - data_test.values[:,idx_soundings_T])**2).mean(axis=0)**0.5

fig = plt.figure(1,(8,8),100)
plt.plot(mean_test,height,'b-*',label = 'ME_LV2_RR')
plt.plot(rmse_test,height,'b-o',label = 'RMSE_LV2_RR')
plt.plot(mean_lv2,height,'k-*',label = 'ME_LV2')
plt.plot(rmse_lv2,height,'k-o',label = 'RMSE_LV2')
plt.plot(mean_lv2_corrected,height,'r-*',label = 'ME_QC_LV2')
plt.plot(rmse_lv2_corrected,height,'r-o',label = 'RMSE_QC_LV2')

plt.legend(bbox_to_anchor = (0.88, -0.08),ncol=3,prop = {'family': 'Times New Roman','size': 11})

plt.xlabel('Error/K',fontsize = 12,family = 'Times New Roman')
plt.ylabel('Height/m',fontsize = 12,family = 'Times New Roman')
plt.title('Error of Temperature',fontsize = 12,family = 'Times New Roman')
plt.tight_layout()
plt.savefig(r'54511/figure/T.jpg')
plt.show()


# In[ ]:


# 绘图 H
mean_test = get_mean_err(model,data_test.values[:,idx_lv1],data_test.values[:,idx_soundings_H],flag= 1)
rmse_test = get_valloss(model,data_test.values[:,idx_lv1],data_test.values[:,idx_soundings_H],flag= 1)
mean_lv2 = (data_test.values[:,idx_lv2_H] - data_test.values[:,idx_soundings_H]).mean(axis=0)
rmse_lv2 = ((data_test.values[:,idx_lv2_H] - data_test.values[:,idx_soundings_H])**2).mean(axis=0)**0.5
mean_lv2_corrected = (data_test.values[:,idx_lv2_H_corrected] - data_test.values[:,idx_soundings_H]).mean(axis=0)
rmse_lv2_corrected = ((data_test.values[:,idx_lv2_H_corrected] - data_test.values[:,idx_soundings_H])**2).mean(axis=0)**0.5

fig = plt.figure(1,(8,8),100)
plt.plot(mean_test,height,'b-*',label = 'ME_LV2_RR')
plt.plot(rmse_test,height,'b-o',label = 'RMSE_LV2_RR')
plt.plot(mean_lv2,height,'k-*',label = 'ME_LV2')
plt.plot(rmse_lv2,height,'k-o',label = 'RMSE_LV2')
plt.plot(mean_lv2_corrected,height,'r-*',label = 'ME_QC_LV2')
plt.plot(rmse_lv2_corrected,height,'r-o',label = 'RMSE_QC_LV2')

plt.legend(bbox_to_anchor = (0.88, -0.08),ncol=3,prop = {'family': 'Times New Roman','size': 11})

plt.xlabel('Error/%',fontsize = 12,family = 'Times New Roman')
plt.ylabel('Height/m',fontsize = 12,family = 'Times New Roman')
plt.title('Error of Humidity',fontsize = 12,family = 'Times New Roman')
plt.tight_layout()
plt.savefig(r'54511/figure/H.jpg')
plt.show()


# In[ ]:


samples_t = deleteBigErr(model,samples_t)

