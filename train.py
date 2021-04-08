# _*_ coding: utf-8 _*_
# @author: anniequ
# @file: main.py
# @time: 2020/11/12 10:41
# @Software: PyCharm
from datetime import datetime
from time import strftime, localtime
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as f

from datapre import VOCSegDataset, classes
from FCN import fcn, VGGNet, label_accuracy_score


def res_record(content):
    with open('./results/result.txt', 'a') as f:
        f.write(content)


def train(epoches=20, show_vgg_params=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    height = 224
    width = 224
    voc_train = VOCSegDataset(True, height, width)
    voc_test = VOCSegDataset(False, height, width)

    train_data = DataLoader(voc_train, batch_size=4, shuffle=True)
    valid_data = DataLoader(voc_test, batch_size=4)
    # 加载预训练的resnet34网络
    model_root = "./model/resnet34-333f7ec4.pth"
    pretrained_net = models.resnet34(pretrained=False)
    pre = torch.load(model_root)
    pretrained_net.load_state_dict(pre)
    # 分类的总数
    num_classes = len(classes)
    # vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    net = fcn(pretrained_net=pretrained_net, num_classes=num_classes).to(device)

    criterion = nn.NLLLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)

    # start training
    # train data
    train_loss = []
    train_acc = []
    train_acc_cls = []
    train_mean_iu = []
    train_fwavacc = []
    # valid data
    eval_loss = []
    eval_acc = []
    eval_acc_cls = []
    eval_mean_iu = []
    eval_fwavacc = []

    print("Start training at ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    res_record("Time:" + strftime("%Y-%m-%d %H:%M:%S", localtime()) + "\n")
    for e in range(epoches):
        _train_loss = 0
        _train_acc = 0
        _train_acc_cls = 0
        _train_mean_iu = 0
        _train_fwavacc = 0

        prev_time = datetime.now()
        net = net.train()
        for img_data, img_label in train_data:
            im = Variable(img_data).to(device)
            lal = Variable(img_label).to(device)

            # 前向传播
            out = net(im)
            out = f.log_softmax(out, dim=1)
            loss = criterion(out, lal)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _train_loss += loss.item()

            # label_pred输出的是21*224*224的向量，对于每一个点都有21个分类的概率
            # 我们取概率值最大的那个下标作为模型预测的标签，然后计算各种评价指标
            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = lal.data.cpu().numpy()

            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                _train_acc += acc
                _train_acc_cls += acc_cls
                _train_mean_iu += mean_iu
                _train_fwavacc += fwavacc

        # 记录当前轮的数据
        train_loss.append(_train_loss / len(train_data))
        train_acc.append(_train_acc / len(voc_train))
        train_acc_cls.append(_train_acc_cls)
        train_mean_iu.append(_train_mean_iu / len(voc_train))
        train_fwavacc.append(_train_fwavacc)

        net = net.eval()

        _eval_loss = 0
        _eval_acc = 0
        _eval_acc_cls = 0
        _eval_mean_iu = 0
        _eval_fwavacc = 0
        for img_data, img_label in valid_data:
            im = Variable(img_data).to(device)
            lal = Variable(img_label).to(device)
            # forward
            out = net(im)
            out = f.log_softmax(out, dim=1)
            loss = criterion(out, lal)
            _eval_loss += loss.item()

            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = lal.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                _eval_acc += acc
                _eval_acc_cls += acc_cls
                _eval_mean_iu += mean_iu
                _eval_fwavacc += fwavacc
        # record the data of current training
        eval_loss.append(_eval_loss / len(valid_data))
        eval_acc.append(_eval_acc / len(voc_test))
        eval_acc_cls.append(_eval_acc_cls)
        eval_mean_iu.append(_eval_mean_iu / len(voc_test))
        eval_fwavacc.append(_eval_fwavacc)

        # print the results of the current training
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        epoch_str = (
        'Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
            e, _train_loss / len(train_data), _train_acc / len(voc_train), _train_mean_iu / len(voc_train),
               _eval_loss / len(valid_data), _eval_acc / len(voc_test), _eval_mean_iu / len(voc_test)))
        time_str = 'Time:{:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(epoch_str + time_str)
        res_record(epoch_str + '\n')
    print("End training at ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    # 记录数据

    # 绘图
    epoch = np.array(range(epoches))
    plt.plot(epoch, train_loss, label="train_loss")
    # plt.plot(epoch, train_loss, label="valid_loss")
    plt.title("loss during training")
    plt.legend()
    plt.grid()
    plt.savefig(r'./results/train_loss.png')
    plt.show()
    # print(train_loss)
    # print(train_acc)
    # print(eval_acc)

    # print(train_mean_iu)
    # print(eval_mean_iu)
    plt.plot(epoch, train_acc, label="train_acc")
    plt.plot(epoch, eval_acc, label="valid_acc")
    plt.title("accuracy during training")
    plt.legend()
    plt.grid()
    plt.savefig(r'./results/acc.png')
    plt.show()

    plt.plot(epoch, train_mean_iu, label="train_mean_iu")
    plt.plot(epoch, eval_mean_iu, label="valid_mean_iu")
    plt.title("mean iu during training")
    plt.legend()
    plt.grid()
    plt.savefig(r'./results/mean_iu.png')
    plt.show()

    # 测试模型性能
    # 保存模型
    PATH = "./model/change_fcn-resnet34.pth"
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    train(50)
