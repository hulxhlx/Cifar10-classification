from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input
import pickle
import os
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sb
import tensorflow.compat.v1 as tf #使用1.0版本的方法
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.manifold import TSNE
from sklearn import decomposition
import numpy as np
from sklearn.svm import LinearSVC
import pandas as pd
#feature extrction google inception:
#transfer the images to a serires of vector, save it, visualize it
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.disable_v2_behavior()
CONV_BLOCK_COUNT = 0  # 用来命名计数卷积编号
INCEPTION_A_COUNT = 0
INCEPTION_B_COUNT = 0
INCEPTION_C_COUNT = 0


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
x_train, x_test = x_train / 255.0, x_test / 255.0

print('x_train shape:', x_train.shape)

nb_filters_reduction_factor = 8
def Cov2dbn(x, nb_filter, kernel_size, activation, strides, padding, data_format="channels_last",kernel_initializer="he_normal",
              ):
    x = Conv2D(nb_filter, kernel_size, strides = strides, data_format = data_format,kernel_initializer = kernel_initializer, padding = padding )(x)
    # x = BatchNormalization(momentum=0.9997, scale=False)(x)
    x = Activation(activation)(x)
    return x
def inception_v4_stem(x):
    # in original inception-v4, conv stride is 2
    x = Cov2dbn(x,32, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='valid')
    x = Cov2dbn(x,32, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='valid')
    x = Cov2dbn(x,64, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='same')

    # in original inception-v4, stride is 2
    a = MaxPooling2D((3, 3), strides=(1, 1), padding="valid", data_format="channels_last")(x)

    # in original inception-v4, conv stride is 2
    b = Cov2dbn(x, 96, (3, 3), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='valid')
    x = merge.concatenate([a, b], axis=-1)

    a = Cov2dbn(x, 64, (1, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    a = Cov2dbn(a, 96, (3, 3), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='valid')
    b = Cov2dbn(x, 64, (1, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    print("ashape")
    print(a.shape)
    b = Cov2dbn(b, 64, (7, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    b = Cov2dbn(b, 64, (1, 7), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    b = Cov2dbn(b, 96, (3, 3), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='valid')
    print("bshape")
    print(b.shape)
    x = merge.concatenate([a, b], axis=-1)

    # in original inception-v4, conv stride should be 2
    a = Cov2dbn(x, 192, (3, 3), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='valid')
    # in original inception-v4, stride is 2
    b = MaxPooling2D((3, 3), strides=(1, 1),  data_format="channels_last", padding='valid')(x)
    x = merge.concatenate([a, b], axis=-1)

    return x

def inception_v4_A(x):
    a = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last")(x)

    a = Cov2dbn(a, 96, (1, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    b = Cov2dbn(x, 96, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='same')

    c = Cov2dbn(x, 64, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    c = Cov2dbn(c, 96, (3, 3), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    d = Cov2dbn(x, 64, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    d = Cov2dbn(d, 96, (3, 3), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    d = Cov2dbn(d, 96, (3, 3), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    x = merge.concatenate([a, b, c, d], axis=-1)

    return x

def inception_v4_reduction_A(x):
    a = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    b = Cov2dbn(x,384, (3, 3), activation="relu", strides=(2, 2),  data_format="channels_last", kernel_initializer="he_normal",
                padding='valid')
    c = Cov2dbn(x,192, (1, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    c = Cov2dbn(c,224, (3, 3), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    c = Cov2dbn(c,256, (3, 3), activation="relu", strides=(2, 2),  data_format="channels_last", kernel_initializer="he_normal",
                padding='valid')

    x = merge.concatenate([a, b, c], axis=-1)

    return x

def inception_v4_B(x):
    a = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    a = Cov2dbn(a,128, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='same')

    b = Cov2dbn(x,384, (1, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    c = Cov2dbn(x,192, (1, 1), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    c = Cov2dbn(c,224, (1, 7), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    c = Cov2dbn(c,256, (1, 7), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')

    d = Cov2dbn(x,192, (1, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    d = Cov2dbn(d,192, (1, 7), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    d = Cov2dbn(d,224, (7, 1), activation="relu", strides=(1, 1), data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    d = Cov2dbn(d,224, (1, 7), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    d = Cov2dbn(d,256, (7, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')

    x = merge.concatenate([a, b, c, d], axis=-1)

    return x

def inception_v4_reduction_B(x):
    a = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    b = Cov2dbn(x,192, (1, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    b = Cov2dbn(b,192, (3, 3), activation="relu", strides=(2, 2),  data_format="channels_last", kernel_initializer="he_normal",
                padding='valid')
    c = Cov2dbn(x,256, (1, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    c = Cov2dbn(c,256, (1, 7), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    c = Cov2dbn(c,256, (7, 1), activation="relu", strides=(1, 1),  data_format="channels_last", kernel_initializer="he_normal",
                padding='same')
    c = Cov2dbn(c,320, (3, 3), activation="relu", strides=(2, 2), data_format="channels_last", kernel_initializer="he_normal",
                padding='valid')

    x = merge.concatenate([a, b, c],axis=-1)

    return x

def inception_v4_C(x):
    a = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    a = Cov2dbn(a, 256  ,(1, 1) , strides=(1, 1), activation='relu',
                      kernel_initializer='he_normal', padding='same')

    b = Cov2dbn(x, 256  ,(1, 1), strides=(1, 1), activation='relu',
                      kernel_initializer='he_normal', padding='same')

    c = Cov2dbn(x, 384  , (1, 1), strides=(1, 1), activation='relu',
                      kernel_initializer='he_normal', padding='same')
    c1 = Cov2dbn(c, 256  , (1, 3), strides=(1, 1), activation='relu',
                       kernel_initializer='he_normal', padding='same')
    c2 = Cov2dbn(c, 256  , (3, 1), strides=(1, 1), activation='relu',
                       kernel_initializer='he_normal', padding='same')

    d = Cov2dbn(x, 384  , (1, 1), strides=(1, 1), activation='relu',
                      kernel_initializer='he_normal', padding='same')
    d = Cov2dbn(d, 448  , (1, 3), strides=(1, 1), activation='relu',
                      kernel_initializer='he_normal', padding='same')
    d = Cov2dbn(d, 512  , (3, 1), strides=(1, 1), activation='relu',
                      kernel_initializer='he_normal', padding='same')
    d1 = Cov2dbn(d, 256  , (3, 1), strides=(1, 1), activation='relu',
                       kernel_initializer='he_normal', padding='same')
    d2 = Cov2dbn(d, 256  , (1, 3), strides=(1, 1), activation='relu',
                       kernel_initializer='he_normal', padding='same')

    x = merge.concatenate([a, b, c1, c2, d1, d2], axis=-1)

    return x
img_rows, img_cols = 32, 32
img_channels = 3

# in original inception-v4, these are 4, 7, 3, respectively
num_A_blocks = 4
num_B_blocks = 7
num_C_blocks = 3

inputs = Input(shape=(img_rows, img_cols, img_channels))

x = inception_v4_stem(inputs)
for i in range(num_A_blocks):
    x = inception_v4_A(x)
x = inception_v4_reduction_A(x)
for i in range(num_B_blocks):
    x = inception_v4_B(x)
x = inception_v4_reduction_B(x)
for i in range(num_C_blocks):
    x = inception_v4_C(x)

x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)

predictions = Dense(10, activation='softmax')(x)




model = Model(inputs=inputs, outputs=predictions)
model.load_weights('cifar10google473-87.h5.')

xtrain_output = model.predict(x_train)
xtest_output = model.predict(x_test)
np.save('googeoutput.npy',xtest_output)
print('save!')
xtrain_output = np.argmax(xtrain_output, axis=1)
xtest_output = np.argmax(xtest_output, axis=1)
c_train = np.argmax(y_train, axis=1)

c_test = np.argmax(y_test, axis=1)
num = 0
for i in range(len(c_train)):
    if int(c_train[i]) == int(xtrain_output[i]):
        num += 1
rate = float(num) / len(c_train)
print(rate)

num = 0
for i in range(len(c_test)):
    if int(c_test[i]) == int(xtest_output[i]):
        num += 1
rate = float(num) / len(c_test)
print(rate)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(xtest_output, c_test)
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
df_c_matrix = pd.DataFrame(cm, index=[clss for clss in class_labels], columns=[clss for clss in class_labels])
sb.heatmap(df_c_matrix, annot=True)
print("Confusion Matrix\n", cm)
plt.show()