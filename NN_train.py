
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


data_others = pd.read_csv(r'samples_others.csv',header= None)
data_201807 = pd.read_csv(r'samples_201807.csv',header= None)


# In[ ]:


# 剔除日期、晴雨指示
idx_bt = [1,2,3] + list(range(5,19))
idx_t = list(range(19,112))
idx_h = list(range(112,205))


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


# 交叉验证
data_train,data_test = train_test_split(data_others.values,test_size = 0,random_state = 10,shuffle = True)


# In[ ]:


# 创建模型
inputs = tf.keras.Input(shape= (17,),name= 'inputs')
hidden_1 = layers.Dense(48,activation='linear',name= 'hidden_1')(inputs)
dropout_1 = layers.Dropout(rate= 0.1,seed= 10,name= 'dropout_1')(hidden_1)
outputs = layers.Dense(93,activation='linear',name= 'outputs')(dropout_1)
# hidden_2 = layers.Dense(32,activation='relu',name= 'hidden_2')(dropout_1)
# dropout_2 = layers.Dropout(rate= 0.1,seed= 10,name= 'dropout_2')(hidden_2)
# outputs = layers.Dense(93,activation='linear',name= 'T/H')(dropout_2)
model = tf.keras.Model(inputs= inputs,outputs= outputs,name= 'model_H')
# 模型摘要
model.summary()

# 模型编译
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= 0.001),
              loss= 'mean_squared_error',
              metrics= [])

# 训练
history = model.fit(x= data_train[:,idx_bt],
                    y= data_train[:,idx_h],
                    batch_size= None,
                    epochs= 4000,
#                     validation_split= 0.1,
                    validation_data=(data_201807.values[:,idx_bt],data_201807.values[:,idx_h]),
                    callbacks= [tf.keras.callbacks.EarlyStopping(patience= 500,restore_best_weights= True)],
                    verbose= 1)


# In[ ]:


# 保存模型
model.save('model_save_H.h5')
# 加载模型
del model
model = tf.keras.models.load_model('model_save_H.h5')


# In[ ]:


# 画图
rmse = get_valloss(model,data_201807.values,idx_bt,idx_h,1)
fig = plt.figure(1,(8,6),100)
plt.plot(rmse,np.arange(1,94),'b-o')
plt.show()

