import os
import json
import cv2
from hand_data_iter.datasets import plot_box, draw_bd_handpose
import random

if __name__ == "__main__":
    path = "./handpose_datasets/"  # 数据集路径
    output_video_path = "output_video.avi"  # 输出视频路径

    # 获取所有图片文件
    img_files = [f for f in os.listdir(path) if f.endswith('.jpg')]  # 只获取jpg图片
    img_files.sort()  # 按文件名排序，确保顺序一致

    # 假设每五张图片合成一个视频
    frame_count = 0
    frame_width, frame_height = 0, 0  # 视频帧的宽高，稍后从第一张图片中获取

    # 打开视频流（这里就不使用摄像头了）
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码
    video_writer = None  # 视频写入器，先设置为空

    # 遍历所有图片，处理每五张图片
    for i in range(0, len(img_files), 5):
        batch_images = img_files[i:i+5]  # 每次取五张图片

        for img_name in batch_images:
            img_path = os.path.join(path, img_name)
            label_path = img_path.replace('.jpg', '.json')  # 假设标签文件与图片同名，扩展名为.json

            if not os.path.exists(label_path):  # 如果标签文件不存在，跳过该图片
                continue

            img_ = cv2.imread(img_path)  # 读取图片文件

            # 获取图片的宽高，第一次获取并设定
            if frame_width == 0 or frame_height == 0:
                frame_width, frame_height = img_.shape[1], img_.shape[0]

            # 创建视频写入器
            if video_writer is None:
                video_writer = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

            # 读取标签文件
            with open(label_path, encoding='utf-8') as f:
                hand_dict_ = json.load(f)  # 将JSON文件内容加载为字典

            hand_dict_ = hand_dict_["info"]  # 获取字典中“info”字段，这个字段包含了手部关键信息
            print("len hand_dict :", len(hand_dict_))  # 输出该图片中包含的手部数据条数

            # 如果存在手部信息，则进行处理
            if len(hand_dict_) > 0:
                for msg in hand_dict_:
                    bbox = msg["bbox"]  # 获取手部边界框坐标
                    pts = msg["pts"]  # 获取手部关键点坐标
                    print(bbox)  # 输出边界框坐标
                    # 随机生成一种颜色，用于绘制边界框
                    RGB = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    # 绘制手部边界框
                    plot_box(bbox, img_, color=RGB, label="hand", line_thickness=3)
                    # 绘制手部关键点连线
                    draw_bd_handpose(img_, pts, bbox[0], bbox[1])

                    # 在图像上绘制每一个手部关键点
                    for k_ in pts.keys():
                        # 关键点的坐标加上边界框的偏移量，确保关键点绘制正确位置
                        cv2.circle(img_, (int(pts[k_]['x'] + bbox[0]), int(pts[k_]['y'] + bbox[1])), 3, (255, 50, 155), -1)

            # 显示当前处理的图片
            cv2.imshow("HandPose_Json", img_)  # 显示处理后的图像

            # 将当前帧写入输出视频流
            video_writer.write(img_)

        # 如果按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频流
    video_writer.release()
    cv2.destroyAllWindows()

    print('Processing finished, video saved to:', output_video_path)
