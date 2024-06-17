'''
Author: BigCiLeng && bigcileng@outlook.com
Date: 2024-06-17 21:14:03
LastEditors: BigCiLeng && bigcileng@outlook.com
LastEditTime: 2024-06-17 21:14:17
FilePath: \R2G\src\scripts\img2video.py
Description: 

Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
'''
import cv2
import os

image_folder = '/DATA_EDS2/luoly/wangn/code/Cutie/examples/images/bike'
video_name = 'output.mp4'

# 获取图片文件夹中的所有图片文件名
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# 排序以确保图片按顺序合并到视频中
images.sort()

# 获取第一张图片的宽度和高度，用于设置视频的分辨率
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape

# 创建视频 writer 对象
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

# 逐帧写入图片到视频
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# 释放资源
cv2.destroyAllWindows()
video.release()