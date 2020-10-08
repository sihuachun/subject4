#!/usr/bin/env python 3.6.7
# -*- coding: utf-8 -*-
# author  : sihuachun
# date    : 2020/10/7
# software: PyCharm

import os
import numpy as np
from cv2 import imread
from collections import OrderedDict
import datetime
import json
import math


def calculate_mean_std(path='C:/Users/sihua/Desktop/cv-competition/subject4-data/images'):
    filepath = path  # 数据集目录
    pathDir = os.listdir(filepath)
    img_size = 1024
    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])

    num = len(pathDir) * img_size * img_size  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

    R_std = np.sqrt(R_channel / num)
    G_std = np.sqrt(G_channel / num)
    B_std = np.sqrt(B_channel / num)
    print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
    print("R_var is %f, G_var is %f, B_var is %f" % (R_std, G_std, B_std))
    mean = (R_mean, G_mean, B_mean)
    std = (R_std, G_std, B_std)
    print(mean, std)
    return mean, std


def transform_annotations(path_images="C:/Users/sihua/Desktop/cv-competition/subject4-data/images", path_labels="C:/Users/sihua/Desktop/cv-competition/subject4-data/labels"):
    pathDir_images = os.listdir(path_images)
    pathDir_labels = os.listdir(path_labels)
    root = "/".join(path_images.split('/')[:-1])
    os.makedirs(root + '/annotations', exist_ok=True)
    information = OrderedDict()
    information["info"] = {"description": "", "url": "", "version": "", "year": 2020, "contributor": "",
                           "date_created": str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))}
    categories = []
    for id in range(1, 6):
        categories.append({"id": id, "name": "ship{}".format(id), "supercategory": "None"})
    information["categories"] = categories
    information["licenses"] = [{"id": 1, "name": None, "url": None}]

    images = []
    H, W, C = imread(os.path.join(path_images, pathDir_images[0])).shape
    for image_name in pathDir_images:
        id = image_name.split('.')[0]
        images.append({"id": int(id), "file_name": image_name, "width": H, "height": W, "date_captured": "", "license": 1, "coco_url": "", "flickr_url": ""})
    information["images"] = images

    annotations = []
    annot_id = 0
    for label_name in pathDir_labels:
        image_id = label_name.split('.')[0]
        with open(os.path.join(path_labels, label_name), mode='r') as f:
            line = f.readline()
            if line:
                line_list = line.strip("\n").strip(" ").strip("\t").split(" ")
                cls, x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line_list))
                x, y = [x1, x2, x3, x4], [y1, y2, y3, y4]
                x_min, x_max = min(x), max(x)
                y_min, y_max = min(y), max(y)
                width, height = x_max - x_min, y_max - y_min
                annotations.append({"id": annot_id, "image_id": int(image_id), "category_id": cls, "iscrowd": 0, "area": width * height, "bbox": [x_min, y_min, width, height], "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]]})
                annot_id += 1
    information["annotations"] = annotations
    with open(os.path.join(root, "annotations", "instances.json"), mode='w+') as f:
        json.dump(information, f)


if __name__ == '__main__':
    calculate_mean_std()
    # transform_annotations()