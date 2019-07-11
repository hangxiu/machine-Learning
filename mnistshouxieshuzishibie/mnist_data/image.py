from PIL import Image

import os

import struct

import numpy as np

# 解析ubyte文件函数


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""

    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)

    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:

        magic, n = struct.unpack('>II', lbpath.read(8))

        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:

        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))

        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


# 矩阵转化为图片函数


def MatrixToImage(data):

    data = data * 255

    new_im = Image.fromarray(data.astype(np.uint8))

    return new_im


# 保存图片函数


def saveToImageQ(image, path, rowNum, number):

    name = str(number) + '.' + str(rowNum)

    # print(os.path.join(path,'%s.jpg'%name))

    image.save(os.path.join(path, '%s.jpg' % name))


# 训练文件所在目录

X_train, y_train = load_mnist('F:\\Data\\mnist\\', kind='train')

for x_ind in range(len(X_train)):

    # 图像提取

    temp_x = X_train[x_ind].reshape(28, 28)

    # 标签提取

    temp_label = y_train[x_ind].reshape(1, 1)

    # 图像标签最后转换

    end_label = ((temp_label.tolist())[0])[0]

    # print(end_label)

    # 将数组转换为图像

    new_im = MatrixToImage(temp_x)

    # 测试50000张图片输出目录

    saveToImageQ(new_im, 'F:\\Data\\mnist\\train50000', x_ind, end_label)

# 测试文件所在目录

X_test, y_test = load_mnist('F:\\Data\\mnist\\', kind='t10k')

for x_ind in range(len(X_test)):

    # 图像提取

    temp_x = X_test[x_ind].reshape(28, 28)

    # 标签提取

    temp_label = y_test[x_ind].reshape(1, 1)

    # 图像标签最后转换

    end_label = ((temp_label.tolist())[0])[0]

    # print(end_label)

    # 将数组转换为图像

    new_im = MatrixToImage(temp_x)

    # 测试10000张图片所在目录

    saveToImageQ(new_im, 'F:\\Data\\mnist\\test10000', x_ind, end_label)
