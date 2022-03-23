from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

pos = 1
i = 0

cifar10_labels = np.array([
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'])

for img in x_train[:50]:
    # plt.subplot(5行10列、サブ領域の番号)
    plt.subplot(5,10,pos)
    plt.imshow(img)
    plt.axis('off')
    plt.title(cifar10_labels[y_train[i][0]])
    pos += 1 
    i += 1
    
plt.show()