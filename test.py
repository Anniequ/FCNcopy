# _*_ coding: utf-8 _*_
# @author: anniequ
# @file: test.py
# @time: 2020/11/17 15:02
# @Software: PyCharm

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.models as models
import torch

from datapre import VOCSegDataset, crop, classes
from FCN import fcn

height = 224
width = 224

voc_test = VOCSegDataset(False, height, width)

PATH = "./model/change_fcn-resnet34.pth"
# 各种标签所对应的颜色
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]]
cm = np.array(COLORMAP).astype('uint8')


def predict(img1, label):
    img1 = Variable(img1.unsqueeze(0)).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载预训练的resnet34网络

    model_root = "./model/resnet34-333f7ec4.pth"
    pretrained_net = models.resnet34(pretrained=False)
    pre = torch.load(model_root)
    pretrained_net.load_state_dict(pre)
    # 分类的总数
    num_classes = len(classes)
    # vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    net = fcn(pretrained_net=pretrained_net, num_classes=num_classes).to(device)

    net.load_state_dict(torch.load(PATH))
    out = net(img1)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    pred = cm[pred]

    pred = Image.fromarray(pred)
    label1 = cm[label.numpy()]
    return pred, label1


SIZE = 224
NUM_IMG = 30
# _, figs = plt.subplots(NUM_IMG, 3, figsize=(12, 22))
for i in range(NUM_IMG):
    img_data, img_label = voc_test[i]
    pred, label = predict(img_data, img_label)
    img_data = Image.open(voc_test.data_list[i])
    img_label = Image.open(voc_test.label_list[i])
    img_data, img_label = crop(img_data, img_label, SIZE, SIZE)
    pred.save("./data/results/"+str(i)+"_pred.png",'PNG')
    img_data.save("./data/results/"+str(i)+"_img.png",'PNG')
    img_label.save("./data/results/"+str(i)+"_label.png",'PNG')
    print("The {}/{} pictures has predicted.".format(i, NUM_IMG))

print("saving predict results finish.")