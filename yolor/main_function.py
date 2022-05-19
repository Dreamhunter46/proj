import argparse
import sys
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
from google.colab.patches import cv2_imshow
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import io
from PIL import Image
import base64
from detect_track_ced import detect
from detect_track import detect_a


from pose import pose_detection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='colab', help='type of machine')
    parser.add_argument('--show', type=str, default='True', help='show on stream webcam')
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort_2.yaml")
    parser.add_argument("--detect_model", type=int, default=0)

    opt = parser.parse_args()
    #print(opt)

    with torch.no_grad():
      bbox, label = detect(opt)
    return bbox, label
