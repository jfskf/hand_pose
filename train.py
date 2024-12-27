# -*-coding:utf-8-*-
# date: 2020-06-24
# Author: Eric.Lee
## function: train

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from utils.model_utils import *
from utils.common_utils import *
from hand_data_iter.datasets import *  # 导入数据集相关模块

from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.squeezenet import squeezenet1_1, squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from models.rexnetv1 import ReXNetV1

from torchvision.models import shufflenet_v2_x1_5, shufflenet_v2_x1_0, shufflenet_v2_x2_0

from loss.loss import *  # 导入损失函数
import cv2
import time
import json
from datetime import datetime


# 定义训练函数
def trainer(ops, f_log):
    try:
        # 设置CUDA设备
        os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

        # 设置日志输出
        if ops.log_flag:
            sys.stdout = f_log

        # 设置随机种子，确保结果可复现
        set_seed(ops.seed)

        # 构建模型
        if ops.model == 'resnet_50':
            model_ = resnet50(pretrained=True, num_classes=ops.num_classes, img_size=ops.img_size[0],
                              dropout_factor=ops.dropout)
        elif ops.model == 'resnet_18':
            model_ = resnet18(pretrained=True, num_classes=ops.num_classes, img_size=ops.img_size[0],
                              dropout_factor=ops.dropout)
        elif ops.model == 'resnet_34':
            model_ = resnet34(pretrained=True, num_classes=ops.num_classes, img_size=ops.img_size[0],
                              dropout_factor=ops.dropout)
        elif ops.model == 'resnet_101':
            model_ = resnet101(pretrained=True, num_classes=ops.num_classes, img_size=ops.img_size[0],
                               dropout_factor=ops.dropout)
        elif ops.model == "squeezenet1_0":
            model_ = squeezenet1_0(pretrained=True, num_classes=ops.num_classes, dropout_factor=ops.dropout)
        elif ops.model == "squeezenet1_1":
            model_ = squeezenet1_1(pretrained=True, num_classes=ops.num_classes, dropout_factor=ops.dropout)
        elif ops.model == "shufflenetv2":
            model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes, dropout_factor=ops.dropout)
        elif ops.model == "shufflenet_v2_x1_5":
            model_ = shufflenet_v2_x1_5(pretrained=False, num_classes=ops.num_classes)
        elif ops.model == "shufflenet_v2_x1_0":
            model_ = shufflenet_v2_x1_0(pretrained=False, num_classes=ops.num_classes)
        elif ops.model == "shufflenet_v2_x2_0":
            model_ = shufflenet_v2_x2_0(pretrained=False, num_classes=ops.num_classes)
        elif ops.model == "shufflenet":
            model_ = ShuffleNet(num_blocks=[2, 4, 2], num_classes=ops.num_classes, groups=3, dropout_factor=ops.dropout)
        elif ops.model == "mobilenetv2":
            model_ = MobileNetV2(num_classes=ops.num_classes, dropout_factor=ops.dropout)
        elif ops.model == "ReXNetV1":
            model_ = ReXNetV1(num_classes=ops.num_classes, dropout_factor=ops.dropout)
        else:
            print("no support the model")  # 不支持的模型

        # 判断是否有可用的GPU
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")  # 使用GPU或CPU
        model_ = model_.to(device)  # 将模型加载到计算设备上

        # 数据集加载
        dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size, flag_agu=ops.flag_agu, fix_res=ops.fix_res,
                                      vis=False)
        print("handpose done")
        print('len train datasets : %s' % (dataset.__len__()))  # 输出训练集样本数量

        # 数据加载器
        dataloader = DataLoader(dataset,
                                batch_size=ops.batch_size,
                                num_workers=ops.num_workers,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)

        # 优化器设置（Adam优化器）
        optimizer_Adam = torch.optim.Adam(model_.parameters(), lr=ops.init_lr, betas=(0.9, 0.99), weight_decay=1e-6)
        optimizer = optimizer_Adam  # 使用Adam优化器

        # 加载微调模型（如果存在）
        if os.access(ops.fintune_model, os.F_OK):  # 检查微调模型路径是否存在
            chkpt = torch.load(ops.fintune_model, map_location=device)
            model_.load_state_dict(chkpt)  # 加载微调模型
            print('load fintune model : {}'.format(ops.fintune_model))

        print('/**********************************************/')

        # 损失函数设置，默认使用MSE损失
        if ops.loss_define != 'wing_loss':
            criterion = nn.MSELoss(reduce=True, reduction='mean')  # 均方误差损失函数

        # 初始化训练参数
        step = 0
        idx = 0
        best_loss = np.inf  # 初始最优损失
        loss_mean = 0.  # 损失均值
        loss_idx = 0.  # 损失计算计数器
        flag_change_lr_cnt = 0  # 学习率更新计数器
        init_lr = ops.init_lr  # 初始学习率
        epochs_loss_dict = {}  # 存储每个epoch的损失值

        # 训练过程
        for epoch in range(0, ops.epochs):
            if ops.log_flag:
                sys.stdout = f_log  # 设置日志输出

            print('\nepoch %d ------>>>' % epoch)
            model_.train()  # 设置模型为训练模式

            # 学习率更新策略
            if loss_mean != 0.:
                if best_loss > (loss_mean / loss_idx):  # 如果当前平均损失更优，更新最优损失
                    flag_change_lr_cnt = 0
                    best_loss = (loss_mean / loss_idx)
                else:
                    flag_change_lr_cnt += 1
                    if flag_change_lr_cnt > 50:  # 连续50次未改善，衰减学习率
                        init_lr = init_lr * ops.lr_decay
                        set_learning_rate(optimizer, init_lr)  # 更新学习率
                        flag_change_lr_cnt = 0

            # 重置损失计数器
            loss_mean = 0.
            loss_idx = 0.

            # 遍历训练数据
            for i, (imgs_, pts_) in enumerate(dataloader):
                if use_cuda:
                    imgs_ = imgs_.cuda()  # 将图片数据送入GPU
                    pts_ = pts_.cuda()  # 将标注数据送入GPU

                output = model_(imgs_.float())  # 模型前向传播
                # 根据定义的损失函数计算损失
                if ops.loss_define == 'wing_loss':
                    loss = got_total_wing_loss(output, pts_.float())  # 使用自定义的wing_loss
                else:
                    loss = criterion(output, pts_.float())  # 默认使用均方误差损失

                loss_mean += loss.item()  # 累加损失
                loss_idx += 1.

                if i % 10 == 0:  # 每10个batch打印一次日志
                    loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print('  %s - %s - epoch [%s/%s] (%s/%s):' % (
                    loc_time, ops.model, epoch, ops.epochs, i, int(dataset.__len__() / ops.batch_size)),
                          'Mean Loss : %.6f - Loss: %.6f' % (loss_mean / loss_idx, loss.item()),
                          ' lr : %.8f' % init_lr, ' bs :', ops.batch_size,
                          ' img_size: %s x %s' % (ops.img_size[0], ops.img_size[1]), ' best_loss: %.6f' % best_loss)

                # 反向传播
                loss.backward()
                optimizer.step()  # 更新优化器参数
                optimizer.zero_grad()  # 清空梯度

                step += 1

            # 保存模型
            torch.save(model_.state_dict(),
                       ops.model_exp + '{}-size-{}-model_epoch-{}.pth'.format(ops.model, ops.img_size[0], epoch))

    except Exception as e:
        print('Exception : ', e)  # 捕获并打印异常
        print('Exception  file : ', e.__traceback__.tb_frame.f_globals['__file__'])  # 异常所在文件
        print('Exception  line : ', e.__traceback__.tb_lineno)  # 异常所在行数


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Hand Train')  # 创建参数解析器
    # 添加参数
    parser.add_argument('--seed', type=int, default=126673, help='seed')  # 随机种子
    parser.add_argument('--model_exp', type=str, default='./model_exp', help='model_exp')  # 模型输出文件夹
    parser.add_argument('--model', type=str, default='ReXNetV1', help='model type')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=42, help='num_classes')  # 类别数量
    parser.add_argument('--GPUS', type=str, default='0', help='GPUS')  # GPU选择
    parser.add_argument('--train_path', type=str, default="./handpose_datasets/", help='datasets')  # 训练数据路径
    parser.add_argument('--pretrained', type=bool, default=True, help='imageNet_Pretrain')  # 是否使用ImageNet预训练权重
    parser.add_argument('--fintune_model', type=str, default='None', help='fintune_model')  # 微调模型路径
    parser.add_argument('--loss_define', type=str, default='wing_loss', help='define_loss')  # 损失函数定义
    parser.add_argument('--init_lr', type=float, default=1e-3, help='init learning Rate')  # 初始学习率
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learningRate_decay')  # 学习率衰减
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')  # 权重衰减
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')  # 动量
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')  # 批次大小
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')  # dropout率
    parser.add_argument('--epochs', type=int, default=3000, help='epochs')  # 训练轮次
    parser.add_argument('--num_workers', type=int, default=10, help='num_workers')  # 数据加载线程数
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='img_size')  # 输入图像尺寸
    parser.add_argument('--flag_agu', type=bool, default=True, help='data_augmentation')  # 是否进行数据增强
    parser.add_argument('--fix_res', type=bool, default=False, help='fix_resolution')  # 是否固定图像分辨率
    parser.add_argument('--clear_model_exp', type=bool, default=False, help='clear_model_exp')  # 是否清除模型输出文件夹
    parser.add_argument('--log_flag', type=bool, default=False, help='log flag')  # 是否保存日志

    # 解析参数
    args = parser.parse_args()

    # 创建模型输出文件夹
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time) + '/'
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)

    # 设置日志输出
    f_log = None
    if args.log_flag:
        f_log = open(args.model_exp + '/train_{}.log'.format(time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)), 'a+')
        sys.stdout = f_log

    # 输出日志信息
    print('---------------------------------- log : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", loc_time)))
    print('\n/******************* {} ******************/\n'.format(parser.description))

    # 将命令行参数保存为字典并输出
    unparsed = vars(args)
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))

    # 保存参数配置
    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)
    fs = open(args.model_exp + 'train_ops.json', "w", encoding='utf-8')
    json.dump(unparsed, fs, ensure_ascii=False, indent=1)
    fs.close()

    # 开始训练
    trainer(ops=args, f_log=f_log)

    if args.log_flag:
        sys.stdout = f_log  # 恢复日志输出
    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))