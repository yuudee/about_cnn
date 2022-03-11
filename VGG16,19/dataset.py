from keras.datasets import cifar10
import keras
from keras.utils import np_utils

def create_data():
    # データの読み込み
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()

    #クラス数、の設定
    num_classes = 10

    # one-hot-vevtor(2値クラスの行列の変換)
    y_train = np_utils.to_categorical(y_train,num_classes)
    y_test = np_utils.to_categorical(y_test,num_classes)
    
    # 形状の表示
    print("x_train : ", x_train.shape)
    print("y_train : ", y_train.shape)
    print("x_test : ", x_test.shape)
    print("y_test : ", y_test.shape)
    
    return x_train,y_train,x_test,y_test

create_data()