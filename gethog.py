import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import pickle
from PIL import Image
#get hog vector and save it!
CIFAR_DIR = "./cifar-10-batches-py"


def load_data(filename):
	'''read data from data file'''
	with open(filename, 'rb') as f:
		data = pickle.load(f, encoding='bytes')  # python3 encoding='bytes'
		return data[b'data'], data[b'labels']


class CifarData:
	def __init__(self, filenames):
		'''参数1:文件夹 参数2:是否需要随机打乱'''
		all_data = []
		all_labels = []

		for filename in filenames:
			data, labels = load_data(filename)
			all_data.append(data)
			all_labels.append(labels)

		self._data = np.vstack(all_data)
		self._data = self._data
		self._labels = np.hstack(all_labels)
		self._num_examples = self._data.shape[0]

	def getdata(self):
		data = self._data.reshape(self._num_examples, 32, 32, 3)
		labels = self._labels
		return data, labels

def gamma(img):
    return np.power(img / 255.0, 1)

def render_gradient( image, cell_gradient):
    cell_size  = 16
    bin_size = 9
    angle_unit = 180// bin_size
    cell_width =  cell_size / 2
    max_mag = np.array(cell_gradient).max()
    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = angle_unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(abs(magnitude))))
                angle += angle_gap
    return image

def div(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x , axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell

def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = np.int8(grad_cell[i, j].flatten())  # .flatten()为降维函数，将其降维为一维，每个cell中的64个梯度值展平，并转为整数
            ang_list = ang_cell[i, j].flatten()  # 每个cell中的64个梯度方向展平
            ang_list = np.int8(ang_list / 20.0)  # 0-9
            ang_list[ang_list >= 9] = 0
            for m in range(len(ang_list)):
                binn[ang_list[m]] += int(grad_list[m])
            bins[i][j] = binn

    return bins

def hog(img, cell_x, cell_y, cell_w):
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y
    gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
    gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)
    print(gradient_magnitude.shape, gradient_angle.shape)
    gradient_angle[gradient_angle > 0] *= 180 / 3.14
    gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + 3.14) * 180 / 3.14
    plt.imshow(gradient_angle ) #显示该图像的梯度方向值
    plt.show()
    grad_cell = div(gradient_magnitude, cell_x, cell_y, cell_w)
    ang_cell = div(gradient_angle, cell_x, cell_y, cell_w)
    bins = get_bins(grad_cell, ang_cell)
    hog_image = render_gradient(np.zeros([img.shape[0], img.shape[1]]), bins)
    plt.imshow(hog_image, cmap=plt.cm.gray)
    plt.show()
    feature = []
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = []
            tmp.append(bins[i, j])
            tmp.append(bins[i + 1, j])
            tmp.append(bins[i, j + 1])
            tmp.append(bins[i + 1, j + 1])
            tmp -= np.mean(tmp)
            feature.append(tmp.flatten())
    return np.array(feature).flatten()


def main():
    cell_w = 8


    train_filename = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
    # 拿到标签
    test_filename = [os.path.join(CIFAR_DIR, 'test_batch')]

    # 拿到训练数据和测试数据
    train_data, train_labels = CifarData(train_filename).getdata()
    test_data, test_labels = CifarData(test_filename).getdata()
    cifar1 = train_data[10]
    img = np.reshape(cifar1, (3, 32, 32))
    img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_name = '1.jpg'
    cv2.imwrite(img_name, img)
    x = img.shape[0] - img.shape[0] % cell_w
    y = img.shape[1] - img.shape[1] % cell_w
    resizeimg = cv2.resize(img, (y, x), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("resizeimg",resizeimg)
    cell_x = int(resizeimg.shape[0] // cell_w)
    cell_y = int(resizeimg.shape[1] // cell_w)  
    gammaimg = gamma(resizeimg) * 255
    feature = hog(gammaimg, cell_x, cell_y, cell_w)
    print(feature.shape)

if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

