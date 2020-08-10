#!/usr/bin/env python
# coding: utf-8

# In[14]:


from tensorflow  import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, MaxPooling2D
from tensorflow.keras import datasets
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)



model = Sequential()

model.add(Conv2D(16,(3,3), activation = 'relu', input_shape = (32, 32, 3) ))
#model.add(MaxPooling2D((2,2), padding = "same"))
model.add(Conv2D(32,(3,3), activation = 'relu' ))
model.add(MaxPooling2D((2,2), padding = "same"))
model.add(Conv2D(64,(3,3), activation = 'relu' ))
model.add(MaxPooling2D((2,2), padding = "same"))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer='adam',
              #loss= tf.keras.losses.categorical_crossentropy(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train,y_train,epochs=8)

model.evaluate(x_test,  y_test)

#model.weights


# In[107]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[85]:


x_train.shape


# In[89]:





# In[90]:


train_images.shape


# In[6]:


y_train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




