import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import SGD

# 入力画像は224*224*3
def AlexNet():
    model = Sequential()
    
    model.add(Conv2D(filters=96,kernel_size=(11,11),strides=4,activation='relu',padding='valid',input_shape=(224,224,3),name='block1_conv1'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),name='block1_max_pool1'))
    model.add(BatchNormalization(name='bn1'))
    
    model.add(Conv2D(filters=256,kernel_size=5,strides=1,activation='relu',padding='valid',name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),name='block2_pool1'))
    model.add(BatchNormalization(name='bn2'))
    
    model.add(Conv2D(filters=96,kernel_size=(3,3),strides=1,padding='valid',activation='relu',name='block3_conv1'))
    
    model.add(Conv2D(filters=96,kernel_size=(3,3),strides=1,padding='valid',activation='relu',name='block4_conv1'))
    
    model.add(Conv2D(filters=96,kernel_size=(3,3),strides=1,padding='valid',activation='relu',name='block5_conv1'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2,name='block5_pool1'))
    model.add(BatchNormalization(name='bn3'))
    
    # 学習時のみ
    model.add(Dropout(rate=0.2,name='drop1'))
    
    model.add(Flatten(name='flatten'))
    
    model.add(Dense(units=4096,activation='relu',name='fc1'))
    # 学習時のみ
    model.add(Dropout(rate=0.2,name='drop2'))
    model.add(Dense(units=4096,activation='relu',name='fc2'))
    
    # softmax部分のunits数はクラス数
    model.add(Dense(units=10,activation='softmax',name='fc3'))
    
    model.compile(optimizer=SGD(lr=0.01),loss='categorical_crossentropy') 
    return model

model = AlexNet()
model.summary() 

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 block1_conv1 (Conv2D)       (None, 54, 54, 96)        34944

 block1_max_pool1 (MaxPoolin  (None, 26, 26, 96)       0
 g2D)

 bn1 (BatchNormalization)    (None, 26, 26, 96)        384

 block2_conv2 (Conv2D)       (None, 22, 22, 256)       614656

 block2_pool1 (MaxPooling2D)  (None, 10, 10, 256)      0

 bn2 (BatchNormalization)    (None, 10, 10, 256)       1024

 block3_conv1 (Conv2D)       (None, 8, 8, 96)          221280

 block4_conv1 (Conv2D)       (None, 6, 6, 96)          83040

 block5_conv1 (Conv2D)       (None, 4, 4, 96)          83040

 block5_pool1 (MaxPooling2D)  (None, 1, 1, 96)         0

 bn3 (BatchNormalization)    (None, 1, 1, 96)          384

 drop1 (Dropout)             (None, 1, 1, 96)          0

 flatten (Flatten)           (None, 96)                0

 fc1 (Dense)                 (None, 4096)              397312

 drop2 (Dropout)             (None, 4096)              0

 fc2 (Dense)                 (None, 4096)              16781312

 fc3 (Dense)                 (None, 10)                40970

=================================================================
Total params: 18,258,346
Trainable params: 18,257,450
Non-trainable params: 896
"""