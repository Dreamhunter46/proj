import argparse
import sys
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import io
from PIL import Image as PIL_Image
import base64

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,LoadColabStreams
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.models import *
from pose import *
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect_a(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, machine, show_webcam, deep_sort_model = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.machine, opt.show, opt.deep_sort_model
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    if machine == 'colab':
        colab_webcam = True 
    else:
        colab_webcam = False
    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    cfg = 'yolor_p6.cfg'
    #cfg = 'setup.cfg'
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])

    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        if True:  
            dataset = LoadColabStreams(img_size=imgsz)
            nr_sources = 1
        else:    
            view_img = True
            dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    poi_id = -1
    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(DeepSort(deep_sort_model, device, max_dist=cfg.DEEPSORT.MAX_DIST,
                		max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                		max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,))
    outputs = [None] * nr_sources




    # Get names and colors
    names = load_classes('coco.names')#model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Initialize Bounding Box Pixel
        if colab_webcam:
            drawing_array = np.zeros([imgsz,imgsz,4], dtype=np.uint8)

        # Process detections
        number_of_person = 0
        id_pred = 0
        id_pred_new = -1
        for i, det in enumerate(pred):  # detections per image
            if webcam and not colab_webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                # pass detections to deepsort
                outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                #print('poi id {:}'.format(poi_id))
                #print('output, {:}'.format(outputs[i]))
                #id_pred_new = -1
                new_idxs = deepsort_list[i].tracker.get_ds_idx()
                if poi_id != -1 and len(outputs[i]) > 0:
                    #print("ds {:}".format(outputs[i][:, 4]))
                    #print("poi_id".format(poi_id))
                    #print(new_idxs)
                    for j in range(len(new_idxs)):
                        if new_idxs[j][1] == poi_id:
                            #print(outputs[i][j, 4])
                            id_pred_new = new_idxs[j][0]
                            break
                
                # Write results
               # id_pred = 0
                for id_pred, (*xyxy, conf, cls) in enumerate(det):
                    if names[int(cls)] != 'person':
                      continue
                    if id_pred_new != -1 and id_pred != id_pred_new:
                        continue
                    if poi_id != -1 and id_pred_new == -1:
                        break
                    #poi_id = outputs[i,id_pred,4]
                    # Check pose
                    if poi_id == -1:
                        up = False
                        masked = np.zeros_like(im0)
                        masked[int(xyxy[1].cpu()):int(xyxy[3].cpu()),int(xyxy[0].cpu()):int(xyxy[2].cpu())] = im0[int(xyxy[1].cpu()):int(xyxy[3].cpu()),int(xyxy[0].cpu()):int(xyxy[2].cpu())]
                        pose_array, up = pose_analysis(masked)

                    if not up and poi_id == -1:
                        continue
                    elif up and poi_id == -1: 
                        if len(outputs[i]) > 0:
                            for k in range(len(new_idxs)):
                                if new_idxs[k][0] == id_pred:
                                    poi_id = new_idxs[k][1]#int(outputs[i][new_idxs[k][1]][4])
                                    print("poi id %i", poi_id)
                    number_of_person += 1
                    #print("ds {:}".format(outputs[i][id_pred_new, 0:4]))
                    #print("yolo {:}".format(xyxy))
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    
                    if colab_webcam and (show_webcam == 'True'): 
                        if poi_id != -1 and id_pred_new != -1:
                            label = '%s %.2f ds id: %s yolo id %s pred new %s' % (names[int(cls)], conf,poi_id, id_pred, id_pred_new)
                            #plot_one_box(det[id_pred_new, :4], drawing_array, label=label, color=colors[int(cls)])

                        else:
                            label = '%s' % (names[int(cls)])
                        plot_one_box(xyxy, drawing_array, label=label, color=colors[int(cls)])
                        if up:
                            drawing_array[:,:,0:3] = drawing_array[:,:,0:3] + pose_array
                            up = False
                        else:
                            drawing_array[:,:,0:3] = drawing_array[:,:,0:3]
                        drawing_array[:,:,3] = (drawing_array.max(axis = 2) > 0 ).astype(int) * 255     # Make tranparent background
                   # id_pred += 1
            if colab_webcam and (show_webcam == 'True'):
                  # Send bounding box bytes back to javascript
                  # print("in")
                  drawing_PIL = PIL_Image.fromarray(drawing_array, 'RGBA')
                  iobuf = io.BytesIO()
                  drawing_PIL.save(iobuf, format='png')
                  drawing_bytes = 'data:image/png;base64,{}'.format((str(base64.b64encode(iobuf.getvalue()), 'utf-8')))
                  vid_cap.stream_data = drawing_bytes
            else:
                # Print time (inference + NMS)
                print('%f person detected %.3f FPS' % (number_of_person, 1/(t2 - t1)))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

