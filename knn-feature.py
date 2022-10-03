from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.manifold import TSNE
from sklearn import decomposition
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import cifar10
import pandas as pd
#knn feature extraction
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

    (train_images, test_images) = normalization(train_images, test_images)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
    #     buffer_size=10000).batch(batch_size)
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    return train_images, train_labels, test_images, test_labels
x_train, y_train, x_test, y_test = load_images()

xtrain_output = np.load('googletrain3.npy')
xtest_output = np.load('googletest3.npy')
xtotal_output = np.concatenate((xtrain_output, xtest_output),axis=0)
clf = KNeighborsClassifier(n_neighbors=6, weights="distance")
c_train = np.argmax(y_train, axis=1)
c_test = np.argmax(y_test, axis=1)

# plt.scatter( X_total_reduced_tsne[:, 0], X_total_reduced_tsne[:, 1], c=c, cmap='tab10' )


clf.fit(xtotal_output[:50000],c_train)
train_result = clf.predict(xtotal_output[:50000])
num = 0
for i in range(len(train_result)):
    if int(train_result[i]) == int(c_train[i]):
        num += 1
rate = float(num) / len(train_result)
print(rate)


# pca = decomposition.PCA(n_components=9)
# X_test_reduced = pca.fit_transform(xtest_output)
# print(X_test_reduced.shape)
# tsne = TSNE(n_components=2)
# X_test_reduced_tsne = tsne.fit_transform(X_test_reduced)
#
predict_result = clf.predict(xtotal_output[50000:])

num = 0
c_test = np.argmax(y_test, axis=1)
# plt.scatter( X_test_reduced_tsne[:, 0], X_test_reduced_tsne[:, 1], c=c_test, cmap='tab10' )

# print(predict_result.shape)
# print(predict_result)
print(c_test)
for i in range(len(predict_result)):
    if int(predict_result[i]) == int(c_test[i]):
        num += 1
rate = float(num) / len(predict_result)
print(rate)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(predict_result, c_test)
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
df_c_matrix = pd.DataFrame(cm, index=[clss for clss in class_labels], columns=[clss for clss in class_labels])
sb.heatmap(df_c_matrix, annot=True)
print("Confusion Matrix\n", cm)
plt.show()