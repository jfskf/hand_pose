# -*-coding:utf-8-*-
# date: 2021-10-5
# Author: Eric.Lee
# function: onnx Inference

# 引入所需的模块
import os, sys
sys.path.append(os.getcwd())  # 将当前工作目录加入系统路径中，方便导入模块
import onnxruntime  # 用于运行ONNX模型的推理引擎
import onnx  # 用于加载和操作ONNX模型的库
import cv2  # 用于图像处理的库
import torch  # 用于深度学习框架（尽管代码中没有使用到torch）
import numpy as np  # 用于数值计算的库
from hand_data_iter.datasets import draw_bd_handpose  # 用于绘制手势关键点的函数

class ONNXModel():
    def __init__(self, onnx_path, gpu_cfg=False):
        """
        初始化ONNX模型
        :param onnx_path: 模型文件路径
        :param gpu_cfg: 是否使用GPU（默认为False，表示使用CPU）
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)  # 加载ONNX模型
        if gpu_cfg:
            self.onnx_session.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])  # 如果gpu_cfg为True，使用CUDA执行提供者
        self.input_name = self.get_input_name(self.onnx_session)  # 获取模型输入的名称
        self.output_name = self.get_output_name(self.onnx_session)  # 获取模型输出的名称
        print("input_name:{}".format(self.input_name))  # 打印输入名称
        print("output_name:{}".format(self.output_name))  # 打印输出名称

    def get_output_name(self, onnx_session):
        """
        获取模型输出的名称
        :param onnx_session: ONNX模型会话
        :return: 输出名称列表
        """
        output_name = []
        for node in onnx_session.get_outputs():  # 遍历所有输出节点
            output_name.append(node.name)  # 添加输出节点的名称到列表
        return output_name

    def get_input_name(self, onnx_session):
        """
        获取模型输入的名称
        :param onnx_session: ONNX模型会话
        :return: 输入名称列表
        """
        input_name = []
        for node in onnx_session.get_inputs():  # 遍历所有输入节点
            input_name.append(node.name)  # 添加输入节点的名称到列表
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        为输入提供数据
        :param input_name: 输入名称
        :param image_numpy: 输入的图像数据
        :return: 一个字典，键为输入名称，值为图像数据
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy  # 将图像数据映射到每个输入名称
        return input_feed

    def forward(self, image_numpy):
        """
        执行推理，传入图像数据进行前向计算
        :param image_numpy: 输入的图像数据
        :return: 模型的输出结果
        """
        input_feed = self.get_input_feed(self.input_name, image_numpy)  # 获取输入数据
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)  # 执行推理
        return output  # 返回推理结果

# 主程序部分
if __name__ == "__main__":
    img_size = 256  # 设置图像的目标大小
    model = ONNXModel(r"shufflenet_v2_x1_5_size-256.onnx")  # 初始化ONNX模型
    path_ = "./image/"  # 图像文件所在路径
    for f_ in os.listdir(path_):  # 遍历路径下所有文件
        img0 = cv2.imread(path_ + f_)  # 读取图像
        img_width = img0.shape[1]  # 获取图像宽度
        img_height = img0.shape[0]  # 获取图像高度
        img = cv2.resize(img0, (img_size, img_size), interpolation=cv2.INTER_CUBIC)  # 将图像缩放到目标尺寸

        img_ndarray = img.transpose((2, 0, 1))  # 将图像从HWC格式转为CHW格式（适配ONNX模型）
        img_ndarray = img_ndarray / 255.  # 将图像像素值归一化到0-1范围
        img_ndarray = np.expand_dims(img_ndarray, 0)  # 扩展维度以适配模型输入格式

        output = model.forward(img_ndarray.astype('float32'))[0][0]  # 执行推理并获取结果
        output = np.array(output)  # 转换为NumPy数组
        print(output.shape[0])  # 打印输出数据的维度

        pts_hand = {}  # 创建一个字典用于存储手部关键点
        for i in range(int(output.shape[0] / 2)):  # 遍历所有手部关键点
            x = (output[i * 2 + 0] * float(img_width))  # 根据比例恢复原图像的x坐标
            y = (output[i * 2 + 1] * float(img_height))  # 根据比例恢复原图像的y坐标

            pts_hand[str(i)] = {}  # 创建关键点字典
            pts_hand[str(i)] = {
                "x": x,
                "y": y,
            }

        draw_bd_handpose(img0, pts_hand, 0, 0)  # 绘制手部关键点的连线

        # 绘制每个关键点的圆圈
        for i in range(int(output.shape[0] / 2)):
            x = (output[i * 2 + 0] * float(img_width))  # 恢复关键点的x坐标
            y = (output[i * 2 + 1] * float(img_height))  # 恢复关键点的y坐标

            # 在图像上绘制圆圈，表示手部关键点
            cv2.circle(img0, (int(x), int(y)), 3, (255, 50, 60), -1)  # 绘制红色实心圆圈
            cv2.circle(img0, (int(x), int(y)), 1, (255, 150, 180), -1)  # 绘制粉色小圆圈

        cv2.namedWindow('image', 0)  # 创建一个显示窗口
        cv2.imshow('image', img0)  # 在窗口中显示处理后的图像
        if cv2.waitKey(600) == 27:  # 等待键盘输入，按下Esc键退出
            break

        cv2.waitKey(0)  # 等待键盘输入，防止窗口关闭
