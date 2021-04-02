# _*_ coding: utf-8 _*_
# @author: anniequ
# @file: datapre.py
# @time: 2020/11/12 11:07
# @Software: PyCharm

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import torchvision.models as models

voc_root = os.path.join("data", "VOC2012")
np.seterr(divide='ignore', invalid='ignore')


# 读取图片
def read_img(root=voc_root, train=True):
    txt_frame = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')

    with open(txt_frame, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
    return data, label


"""
#数据显示

img = Image.open(data[0])
plt.subplot(2,2,1), plt.imshow(img)

img = Image.open(label[0])
plt.subplot(2,2,2), plt.imshow(img)

img = Image.open(data[1])
plt.subplot(2,2,3), plt.imshow(img)

img = Image.open(label[1])
plt.subplot(2,2,4), plt.imshow(img)
plt.show()

"""


# 图片大小不同，同时裁剪data and label
def crop(data, label, height, width):
    'data and label both are Image object'
    box = (0, 0, width, height)
    data = data.crop(box)
    label = label.crop(box)
    return data, label


"""
# 裁剪后图片显示
img = Image.open(data[0])
lab = Image.open(label[0])
plt.subplot(2,2,1), plt.imshow(img)
plt.subplot(2,2,2), plt.imshow(lab)

img, lab = crop(img, lab, 224, 224)
plt.subplot(2,2,3),plt.imshow(img)
plt.subplot(2,2,4),plt.imshow(lab)
plt.show()
"""

# VOC数据集中对应的标签
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# 各种标签所对应的颜色
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
cm2lbl = np.zeros(256 ** 3)

# 枚举的时候i是下标，cm是一个三元组，分别标记了RGB值
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


# 将标签按照RGB值填入对应类别的下标信息
def image2label(im):
    data = np.array(im, dtype="int32")
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype="int64")


"""
# 标签对应图片
im = Image.open(label[20]).convert("RGB")
label_im = image2label(im)
plt.imshow(im)
plt.show()
print(label_im[100:110, 200:210])

"""


def image_transforms(data, label, height, width):
    data, label = crop(data, label, height, width)
    # 将数据转换成tensor，并且做标准化处理
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = im_tfs(data)
    label = image2label(label)
    label = torch.from_numpy(label)
    return data, label


class VOCSegDataset(torch.utils.data.Dataset):

    # 构造函数
    def __init__(self, train, height, width, transforms=image_transforms):
        self.height = height
        self.width = width
        self.fnum = 0  # 用来记录被过滤的图片数
        self.transforms = transforms
        data_list, label_list = read_img(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        if train == True:
            print("训练集：加载了 " + str(len(self.data_list)) + " 张图片和标签" + ",过滤了" + str(self.fnum) + "张图片")
        else:
            print("测试集：加载了 " + str(len(self.data_list)) + " 张图片和标签" + ",过滤了" + str(self.fnum) + "张图片")

    # 过滤掉长小于height和宽小于width的图片
    def _filter(self, images):
        img = []
        for im in images:
            if (Image.open(im).size[1] >= self.height and
                    Image.open(im).size[0] >= self.width):
                img.append(im)
            else:
                self.fnum = self.fnum + 1
        return img

    # 重载getitem函数，使类可以迭代
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.height, self.width)
        return img, label

    def __len__(self):
        return len(self.data_list)
