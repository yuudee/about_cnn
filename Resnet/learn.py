from resnet import cifar10_dataset,resnet

class Trainer:
    # パラメータの設定
    def __init__(self,model,loss,optimizer):
        self.model = model
        self.model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
        self.verbose = 1
        self.batch_size = 128
        self.epochs = 30
        
    # 学習
    def fit(self,x_train,y_train,x_test,y_test):
        self.model.fit(x_train,y_train,batch_size=self.batch_size,epochs=self.epochs,verbose=self.verbose,validation_data=(x_test,y_test))
        
        return self.model
    
    
dataset = cifar10_dataset()
model = resnet(dataset.image_shape,dataset.num_classes,4)
x_train,y_train,x_test,y_test = dataset.get_batch()
trainer = Trainer(model,loss='categorical_crossentropy',optimizer='adam')
model = trainer.fit(x_train,y_train,x_test,y_test)

# 評価
score = model.evaluate(x_test,y_test,verbose=0)
print('Test loss: ',score[0])
print('Test accuracy: ',score[1])


# categoryorical_crossentropyはone-hotの符号化対象に対して動作
# sparse_categorical_crossentropyは整数対象に対して動作