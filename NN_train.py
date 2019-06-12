
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
# import re
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress = True)


# In[138]:


# 汇总处理
data = pd.read_csv(r'F:\data\beijing2018\LV\train\samples.csv',header= None)

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
data.iloc[idx_201807].to_csv(r'F:\data\beijing2018\LV\train\samples_201807.csv',index = None,header = None)
data.iloc[idx_others].to_csv(r'F:\data\beijing2018\LV\train\samples_others.csv',index = None,header = None)


# In[139]:


data = pd.read_csv(r'F:\data\beijing2018\LV\train\samples_others.csv',header= None)


# In[140]:


# 剔除日期、晴雨指示
idx_bt = [1,2,3] + list(range(5,19))
idx_t = list(range(19,112))
idx_h = list(range(112,205))


# In[210]:


# 交叉验证
data_train,data_test = train_test_split(data.values,test_size = 0.1,random_state = 10,shuffle = True)

# 创建模型
inputs = tf.keras.Input(shape= (17,),name= 'bt')
hidden_1 = layers.Dense(32,activation='relu',name= 'hidden_1')(inputs)
dropout_1 = layers.Dropout(rate= 0.1,seed= 10,name= 'dropout_1')(hidden_1)
outputs = layers.Dense(93,activation='linear',name= 'T/H')(dropout_1)
# hidden_2 = layers.Dense(32,activation='relu',name= 'hidden_2')(dropout_1)
# dropout_2 = layers.Dropout(rate= 0.1,seed= 10,name= 'dropout_2')(hidden_2)
# outputs = layers.Dense(93,activation='linear',name= 'T/H')(dropout_2)
model = tf.keras.Model(inputs= inputs,outputs= outputs,name= 'mymodel')
# 模型摘要
model.summary()

# 模型编译
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= 0.001),
              loss= tf.losses.MeanSquaredError(),
              metrics= [])

# 训练
history = model.fit(x= data_train[:,idx_bt],
                    y= data_train[:,idx_t],
                    batch_size= None,
                    epochs= 500,
#                     validation_split= 0.1,
                    validation_data=(data_test[:,idx_bt],data_test[:,idx_t]),
                    verbose= 1)

