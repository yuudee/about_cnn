from tensorflow.keras import optimizers
from vgg16 import vgg16
from dataset import create_data


'''バッチサイズ、クラス数、エポック数の設定'''
batch_size=64
num_classes=10
epochs=20

model = vgg16()
x_train,y_train,x_test,y_test = create_data()

# optimizerの定義
optimizer = optimizers.Adam()
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# データの正規化
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255

history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))
model.save('vgg16.h5')