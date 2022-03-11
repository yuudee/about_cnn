import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
from dataset import create_data

x_train,y_train,x_test,y_test = create_data()
num_classes = 10


# VGG16の論文ではbarchnormalizationは採用されていない(当時その手法は確立されていない)
def vgg16():
    # x_trainの形状は(画像の枚数、縦、横、次元数)
    input_shape = x_train.shape[1:]
    model = Sequential()
    # filters:フィルター数、paddingには'valid'、'same'がある。
    # validは出力サイズが入力サイズよりも小さくなる。　sameはサイズは同じになる。
    # 1ブロック
    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',input_shape=input_shape,name='block1_conv1'))
    model.add(BatchNormalization(name='bn1'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',name='block1_conv2'))
    model.add(BatchNormalization(name='bn2'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='block1_pool'))

    # 2ブロック
    model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',name='block2_conv1'))
    model.add(BatchNormalization(name='bn3'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',name='block2_conv2'))
    model.add(BatchNormalization(name='bn4'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='block2_pool'))

    # 3ブロック
    model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',name='block3_conv1'))
    model.add(BatchNormalization(name='bn5'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',name='block3_conv2'))
    model.add(BatchNormalization(name='bn6'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',name='block3_conv3'))
    model.add(BatchNormalization(name='bn7'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',name='block3_conv4'))
    model.add(BatchNormalization(name='bn8'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='block3_pool'))

    # 4ブロック
    model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='block4_conv1'))
    model.add(BatchNormalization(name='bn9'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='block4_conv2'))
    model.add(BatchNormalization(name='bn10'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='block4_conv3'))
    model.add(BatchNormalization(name='bn11'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='block4_conv4'))
    model.add(BatchNormalization(name='bn12'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='block4_pool'))

    # 5ブロック
    model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='block5_conv1'))
    model.add(BatchNormalization(name='bn13'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='block5_conv2'))
    model.add(BatchNormalization(name='bn14'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='block5_conv3'))
    model.add(BatchNormalization(name='bn15'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='block5_conv4'))
    model.add(BatchNormalization(name='bn16'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='block5_pool'))

    # 4次元配列を1次元配列に変換
    model.add(Flatten(name='flatten'))

    # 全結合層*3(units:出力数)
    model.add(Dense(units=4096,activation='relu',name='fc1'))
    model.add(Dense(units=4093,activation='relu',name='fc2'))
    model.add(Dense(units=1000,activation='relu',name='fc3'))
    model.add(Dense(units=num_classes,activation='softmax',name='predictions'))
    
    return model

'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792

 bn1 (BatchNormalization)    (None, 32, 32, 64)        256

 activation (Activation)     (None, 32, 32, 64)        0

 block1_conv2 (Conv2D)       (None, 32, 32, 64)        36928

 bn2 (BatchNormalization)    (None, 32, 32, 64)        256

 activation_1 (Activation)   (None, 32, 32, 64)        0

 block1_pool (MaxPooling2D)  (None, 16, 16, 64)        0

 block2_conv1 (Conv2D)       (None, 16, 16, 128)       73856

 bn3 (BatchNormalization)    (None, 16, 16, 128)       512

 activation_2 (Activation)   (None, 16, 16, 128)       0

 block2_conv2 (Conv2D)       (None, 16, 16, 128)       147584

 bn4 (BatchNormalization)    (None, 16, 16, 128)       512

 activation_3 (Activation)   (None, 16, 16, 128)       0

 block2_pool (MaxPooling2D)  (None, 8, 8, 128)         0

 block3_conv1 (Conv2D)       (None, 8, 8, 256)         295168

 bn5 (BatchNormalization)    (None, 8, 8, 256)         1024

 activation_4 (Activation)   (None, 8, 8, 256)         0

 block3_conv2 (Conv2D)       (None, 8, 8, 256)         590080

 bn6 (BatchNormalization)    (None, 8, 8, 256)         1024

 activation_5 (Activation)   (None, 8, 8, 256)         0

 block3_conv3 (Conv2D)       (None, 8, 8, 256)         590080

 bn7 (BatchNormalization)    (None, 8, 8, 256)         1024

 activation_6 (Activation)   (None, 8, 8, 256)         0

 block3_conv4 (Conv2D)       (None, 8, 8, 256)         590080

 bn8 (BatchNormalization)    (None, 8, 8, 256)         1024

 activation_7 (Activation)   (None, 8, 8, 256)         0

 block3_pool (MaxPooling2D)  (None, 4, 4, 256)         0

 block4_conv1 (Conv2D)       (None, 4, 4, 512)         1180160

 bn9 (BatchNormalization)    (None, 4, 4, 512)         2048

 activation_8 (Activation)   (None, 4, 4, 512)         0

 block4_conv2 (Conv2D)       (None, 4, 4, 512)         2359808

 bn10 (BatchNormalization)   (None, 4, 4, 512)         2048

 activation_9 (Activation)   (None, 4, 4, 512)         0

 block4_conv3 (Conv2D)       (None, 4, 4, 512)         2359808

 bn11 (BatchNormalization)   (None, 4, 4, 512)         2048

 activation_10 (Activation)  (None, 4, 4, 512)         0

 block4_conv4 (Conv2D)       (None, 4, 4, 512)         2359808

 bn12 (BatchNormalization)   (None, 4, 4, 512)         2048

 activation_11 (Activation)  (None, 4, 4, 512)         0

 block4_pool (MaxPooling2D)  (None, 2, 2, 512)         0

 block5_conv1 (Conv2D)       (None, 2, 2, 512)         2359808

 bn13 (BatchNormalization)   (None, 2, 2, 512)         2048

 activation_12 (Activation)  (None, 2, 2, 512)         0

 block5_conv2 (Conv2D)       (None, 2, 2, 512)         2359808

 bn14 (BatchNormalization)   (None, 2, 2, 512)         2048

 activation_13 (Activation)  (None, 2, 2, 512)         0

 block5_conv3 (Conv2D)       (None, 2, 2, 512)         2359808

 bn15 (BatchNormalization)   (None, 2, 2, 512)         2048

 activation_14 (Activation)  (None, 2, 2, 512)         0

 block5_conv4 (Conv2D)       (None, 2, 2, 512)         2359808

 bn16 (BatchNormalization)   (None, 2, 2, 512)         2048

 activation_15 (Activation)  (None, 2, 2, 512)         0

 block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0

 flatten (Flatten)           (None, 512)               0

 fc1 (Dense)                 (None, 4096)              2101248

 fc2 (Dense)                 (None, 4093)              16769021

 fc3 (Dense)                 (None, 1000)              4094000

 predictions (Dense)         (None, 10)                10010

=================================================================
Total params: 43,020,679
Trainable params: 43,009,671
Non-trainable params: 11,008
'''