# -*-coding:utf-8-*-
# date: 2021-10-5
# Author: Eric.Lee
# function: pytorch model 2 onnx

import os
import argparse
import torch
import torch.nn as nn
import numpy as np

# 导入不同的模型结构，包括ResNet、SqueezeNet、ShuffleNetV2、MobileNetV2等
from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.squeezenet import squeezenet1_1, squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5, shufflenet_v2_x1_0, shufflenet_v2_x2_0
from models.rexnetv1 import ReXNetV1

if __name__ == "__main__":

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Project Hand Pose Inference')
    parser.add_argument('--model_path', type=str, default='./weights1/resnet_50-size-256-wingloss102-0.119.pth',
                        help='model_path')  # 模型路径
    parser.add_argument('--model', type=str, default='shufflenet_v2_x1_5',
                        help='''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,
                        shufflenet,mobilenetv2 shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0''')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=42,
                        help='num_classes')  # 42个关键点(x, y) * 2 = 42
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')  # 选择使用的GPU
    parser.add_argument('--test_path', type=str, default='./image/',
                        help='test_path')  # 测试图片路径
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')  # 输入模型的图片尺寸
    parser.add_argument('--vis', type=bool, default=True,
                        help='vis')  # 是否可视化图片

    # 输出程序描述信息
    print('\n/******************* {} ******************/\n'.format(parser.description))

    # 解析命令行参数
    ops = parser.parse_args()
    print('----------------------------------')

    # 将参数字典化并打印
    unparsed = vars(ops)
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))

    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    test_path = ops.test_path  # 测试图片文件夹路径

    # 构建模型
    print('use model : %s' % (ops.model))

    # 根据输入的模型类型选择对应的模型
    if ops.model == 'resnet_50':
        model_ = resnet50(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_18':
        model_ = resnet18(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_ = resnet34(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_101':
        model_ = resnet101(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == "squeezenet1_0":
        model_ = squeezenet1_0(num_classes=ops.num_classes)
    elif ops.model == "squeezenet1_1":
        model_ = squeezenet1_1(num_classes=ops.num_classes)
    elif ops.model == "shufflenetv2":
        model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_5":
        model_ = shufflenet_v2_x1_5(num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_0":
        model_ = shufflenet_v2_x1_0(num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x2_0":
        model_ = shufflenet_v2_x2_0(num_classes=ops.num_classes)
    elif ops.model == "shufflenet":
        model_ = ShuffleNet(num_blocks=[2, 4, 2], num_classes=ops.num_classes, groups=3)
    elif ops.model == "mobilenetv2":
        model_ = MobileNetV2(num_classes=ops.num_classes)

    # 判断是否有GPU可用，选择设备
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # 将模型移到对应的设备（GPU/CPU）
    model_ = model_.to(device)

    # 设置模型为评估模式，关闭Dropout等
    model_.eval()

    # 加载测试模型的权重
    if os.access(ops.model_path, os.F_OK):  # 如果模型路径存在
        chkpt = torch.load(ops.model_path, map_location=device)  # 加载模型权重
        model_.load_state_dict(chkpt)  # 将权重加载到模型中
        print('load test model : {}'.format(ops.model_path))

    # 设置输入图像尺寸和批处理大小
    input_size = ops.img_size[0]
    batch_size = 1  # 批处理大小
    input_shape = (3, input_size, input_size)  # 输入数据的形状，通常是(3, H, W)对于RGB图像
    print("input_size : ", input_size)

    # 生成一个随机输入张量，用于模型的导出
    x = torch.randn(batch_size, *input_shape)  # 生成随机输入，形状为(batch_size, 3, height, width)
    x = x.to(device)  # 将输入移到设备上

    # 设置导出的ONNX文件名
    export_onnx_file = "{}_size-{}.onnx".format(ops.model, input_size)

    # 导出模型为ONNX格式
    torch.onnx.export(model_,
                      x,
                      export_onnx_file,
                      opset_version=9,  # 设置ONNX的opset版本
                      do_constant_folding=True,  # 是否进行常量折叠优化
                      input_names=["input"],  # 输入的名字
                      output_names=["output"],  # 输出的名字
                      # dynamic_axes={"input": {0: "batch_size"},  # 如果需要，可以启用动态轴
                      #               "output": {0: "batch_size"}}
                      )

    # 打印导出的ONNX文件路径
    print("导出 ONNX 模型到: ", export_onnx_file)

    # 第二次导出（此行是多余的，可以去掉）
    torch.onnx.export(model_,
                      x,
                      export_onnx_file,
                      opset_version=9,  # 设置ONNX的opset版本
                      do_constant_folding=True,  # 是否进行常量折叠优化
                      input_names=["input"],  # 输入的名字
                      output_names=["output"],  # 输出的名字
                      # dynamic_axes={"input": {0: "batch_size"},  # 如果需要，可以启用动态轴
                      #               "output": {0: "batch_size"}}
                      )

    # 打印最终导出成功的消息
    print(f"模型成功导出为 ONNX 格式: {export_onnx_file}")
