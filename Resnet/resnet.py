import keras
from keras.utils import np_utils
from keras.layers import Input,Activation,Concatenate,Conv2D,BatchNormalization,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Model
from keras.datasets import cifar10

#cifar10のデータを取得してベクトルに変換
class cifar10_dataset():
    # コンストラクタ
    def __init__(self):
        self.image_shape = (32,32,3)
        self.num_classes = 10
     
    # 目的変数の場合は、クラスベクトルに変更する。説明変数は標準化する。 
    def change_vector(self,img_data,label=False):
        if label == True:
            # one-hotベクトルに変換
            img_data = np_utils.to_categorical(img_data,self.num_classes)
        else:
            img_data = img_data.astype('float32')
            # 0から1で正規化
            img_data /= 255
            shape = (img_data.shape[0],) + self.image_shape
            img_data = img_data.reshape(shape)
        return img_data
    
    def get_batch(self):
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()
        x_train,x_test = [self.change_vector(img_data) for img_data in [x_train,x_test]]
        y_train,y_test = [self.change_vector(img_data,label=True) for img_data in [y_train,y_test]]
        
        return x_train,y_train,x_test,y_test
    
def resnet(input_shape,num_classes,count):
    # Conv2d(filters,kernel_size,padding,activation,name)
    filter_count = 32
    inputs = Input(shape=input_shape)
    x = Conv2D(32,kernel_size=3,padding='same',activation='relu')(inputs)
    x = BatchNormalization()(x)
    for i in range(count):
        # 残差接続のため入力データを取得
        shortcut = x
        x = Conv2D(filter_count,kernel_size=3,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.3)(x)
        x = Conv2D(filter_count,kernel_size=3,padding='same')(x)
        x = BatchNormalization()(x)
        # スキップ結合
        x = Concatenate()([x,shortcut])
        
        # 最終層の時
        if i != count -1:
            x = MaxPooling2D(pool_size=2)(x)
            filter_count = filter_count * 2 
            
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(num_classes,activation='softmax')(x)
    model = Model(inputs=inputs,outputs=x)
    print(model.summary())
        
    return model
        
            