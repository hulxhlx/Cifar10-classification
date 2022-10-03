import torch

from keras.datasets import cifar10
import numpy as np

import pickle
import os
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sb
from sklearn.manifold import TSNE
from sklearn import decomposition
import numpy as np
from sklearn.svm import LinearSVC
import pandas as pd
import torchvision
#list all the error and their probability to compare. the input is predict softmax npy file




def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images

def load_images():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    # (train_images, test_images) = normalization(train_images, test_images)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
    #     buffer_size=10000).batch(batch_size)
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    return train_images, train_labels, test_images, test_labels
x_train, y_train, x_test, y_test = load_images()
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


# x_train, x_test = x_train / 255.0, x_test / 255.0




# model = Model(inputs=inputs, outputs=predictions)
# model.load_weights('cifar10google473-87.h5.')
# xtest_output = model.predict(x_test)
# np.save('googeoutput.npy',xtest_output)

test_predict= np.load('respred.npy')

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

y_predict_labels = np.argmax(test_predict, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
confuse_list = []
suspect_list= []
gap = []

typeerr = []
for i in range(10):
    typeerr.append([])
totalerr_list = []
for i in range(10):
    totalerr_list.append([])
d=0
for i in range(len(y_test)):
    if y_test_labels[i] != y_predict_labels[i]:
        if (test_predict[i][y_predict_labels[i]]-test_predict[i][y_test_labels[i]]) <0.1:
            confuse_list.append((test_images[i], y_test_labels[i], y_predict_labels[i],test_predict[i]))
        if (test_predict[i][y_predict_labels[i]]-test_predict[i][y_test_labels[i]]) >0:
            totalerr_list[y_test_labels[i]].append((test_images[i], y_test_labels[i], y_predict_labels[i],test_predict[i]))
        else:
            suspect_list.append((test_images[i], y_test_labels[i], y_predict_labels[i],test_predict[i]))
        gap.append(test_predict[i][y_predict_labels[i]]-test_predict[i][y_test_labels[i]])
        typeerr[y_test_labels[i]].append(test_predict[i][y_predict_labels[i]]-test_predict[i][y_test_labels[i]])
        d=d+1
for i in range(10):
    plt.title(class_labels[i])
    plt.hist(typeerr[i], bins=50, rwidth=0.9, density=False)
    plt.xlabel('prediction gap')
    plt.ylabel('number of samples')
    plt.show()
    print("the label "+class_labels[i]+' has '+str(len(typeerr[i]))+" errors")

plt.title('total samples')
plt.hist(gap, bins=20, rwidth=0.9, density=False)
print(len(confuse_list))
print(len(totalerr_list))
print(len(suspect_list))
plt.xlabel('prediction gap')
plt.ylabel('number of samples')
plt.show()
plt.close()
fig, axs = plt.subplots(10, 5, figsize=(20, 30))
w = 0
print(len(totalerr_list[9]))
for p in range(10):
    w = 0
    for i in range(10):
        for j in range(5):
            image = totalerr_list[p][w][0]
            image = image.reshape(32,32,3)
            label = totalerr_list[p][w][1]
            predict_label = totalerr_list[p][w][2]
            prediction = totalerr_list[p][w][3]
            axs[i, j].imshow(image)
            axs[i, j].set_title(f'True: {class_labels[label]}' \
                            f' Prob: {"%.3f"%prediction[label]}\n'
                            f'Pred: {class_labels[predict_label]}'
                            f' Prob: {"%.3f"%prediction[predict_label]}'
                            ,fontsize=20)
            axs[i, j].axis('off')
            w += 1
            if w+1 > len(totalerr_list[p]):
                break
        if w +1 > len(totalerr_list[p]):
            break
    plt.show()
    plt.close()
    fig, axs = plt.subplots(10, 5, figsize=(20, 30))

#
# for i in range(15):
#     for j in range(10):
#         image = totalerr_list[w][0]
#         image = image.reshape(32,32,3)
#         # image = image.transpose((1, 2, 0))
#         label = totalerr_list[w][1]
#         predict_label = totalerr_list[w][2]
#         prediction = totalerr_list[w][3]
#         axs[i, j].imshow(image)
#         print()
#         axs[i, j].set_title(f'Label: {class_labels[label]}' \
#                             f' Prob: {"%.3f"%prediction[label]}\n'
#                             f'Pred: {class_labels[predict_label]}'
#                             f' Prob: {"%.3f"%prediction[predict_label]}'
#                             ,fontsize=20)
#         axs[i, j].axis('off')
#         w += 1
# plt.show()



# print(classification_report(y_test_labels, y_predict_labels))
