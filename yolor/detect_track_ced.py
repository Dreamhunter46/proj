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

def detect(opt, save_img=False):
    print('detext_track.py')
    
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
    
    # List to monitor FPS of the algo
    yolo_fps = []
    pose_fps = []
    ds_fps = []
    print_fps = []
    total_fps_no_print = []
    total_fps_print = []
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
    poi_detected, deep_sort_already_init, poi_index, poi_index_base = False, False, None, None
    is_poi = False

    for path, img, im0s, vid_cap in dataset:
        is_poi = False
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t0_yolo = time_synchronized()

        # Initialize Bounding Box Pixel
        if colab_webcam:
            drawing_array = np.zeros([imgsz,imgsz,4], dtype=np.uint8)

        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t1_yolo = time_synchronized()
        yolo_fps.append(1/(t1_yolo - t0_yolo))
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Can start going through detections
        i, det = 0, pred[0]
        
        if webcam and not colab_webcam:  # batch_size >= 1
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
        else:
            p, s, im0 = path, '', im0s

        is_detection = False
        if det is not None:
          det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
          is_detection = True
        
        #print(det)
        index = 0 # To keep track of the current bbox being pose-analyzed
        if is_detection:
          for *xyxy, conf, cls in det:

            if names[int(cls)] != 'person':
              continue
            
            # POSE DETECTION #####################################################
            t0_pose = time_synchronized()
            if not poi_detected: # No one detected
              up = False
              masked = np.zeros_like(im0)
              masked[int(xyxy[1].cpu()):int(xyxy[3].cpu()),int(xyxy[0].cpu()):int(xyxy[2].cpu())] = im0[int(xyxy[1].cpu()):int(xyxy[3].cpu()),int(xyxy[0].cpu()):int(xyxy[2].cpu())]
              pose_array, up = pose_analysis(masked)
              # print(det)
              if not up:
                index += 1 # Go to next bbox
                continue
              if up:
                poi_index_yolo = index # Save index in the list to be given to DeepSort
                poi_detected = True # Do not check pose again
                drawing_array[:,:,0:3] = drawing_array[:,:,0:3] + pose_array
                t1_pose = time_synchronized()
                pose_fps.append(1/(t1_pose - t0_pose))
                pass # get out of the for loop

            # END OF POSE DETECTION ##############################################

            # If up, save bbox and feed it through DeepSort
            # So, only take care of bbox at "index"

          # Save bboxs, confidences and classes
          xywhs = xyxy2xywh(det[:, 0:4])
          confs = det[:, 4]
          clss = det[:, 5]
          
          # print(confs)
          # print(clss)
          # print(xywhs)

        if poi_detected and is_detection: # Person was already detected

        # DeepSort tracking ####################################################
          t0_ds = time_synchronized()
          if not deep_sort_already_init:
            outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            if len(outputs[i]) > 0: # First time getting up to here
              deep_sort_already_init = True # Do not come again
              #print('only once: outputs[i]: {}'.format(outputs[i]))
              poi_index = outputs[i][poi_index_yolo][4] # Retrieve DeepSort poi index
              #print('Deepsort poi index {}'.format(poi_index))
              #for output in outputs[i]: # Go through DeepSort detections
              #  if output[4] == poi_index:
              bbox = det[poi_index_yolo,:4] # Bbox from yolo
              conf = det[poi_index_yolo,4]
              clss = det[poi_index_yolo,5]
              bbox_position_x = (bbox[0] + bbox[2]) / 2
              bbox_position_y = (bbox[1] + bbox[3]) / 2
              time_pos = time.time()
          else: # Already got DeepSort poi index
            outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            #print('every time: outputs[i]: {}'.format(outputs[i]))
            #[(track_id, det_id),...]
            new_idxs = deepsort_list[i].tracker.get_ds_idx()
            #print("id", new_idxs)
            #print("out", outputs[i])
            new_idxs_ut = deepsort_list[i].tracker.get_ut_ds_idx()
            if len(outputs[i]) > 0:
              # print('PoI ID %i'%outputs[i][4])
              for num, indices in enumerate(outputs[i]):
                if names[int(indices[5])] == 'person' and indices[4] == poi_index:
                  for k in range(len(new_idxs)):
                    if new_idxs[k][0] == num:
                      is_poi = True
                      #print("detected")
                      poi_index_yolo = new_idxs[k][1] # Poi index in the yolo list
                      bbox = det[poi_index_yolo, :4]#output[:4] # Bbox from yolo
                      conf = det[poi_index_yolo, 4] #output[4]
                      clss = det[poi_index_yolo, 5] #output[5]
                      
                      # Compute center position and speed
                      last_pos_x = bbox_position_x
                      last_pos_y = bbox_position_y
                      bbox_position_x = (bbox[0] + bbox[2]) / 2
                      bbox_position_y = (bbox[1] + bbox[3]) / 2
                      width_x = abs(bbox[2] - bbox[0])
                      height_y = abs(bbox[3] - bbox[1])
                      if time_pos:
                        delta_time = time.time() - time_pos
                        bbox_speed_x = (bbox_position_x - last_pos_x) / delta_time
                        bbox_speed_y = (bbox_position_y - last_pos_y) / delta_time
                        time_pos = time.time()
                      print('Pos. x {}, Pos. y {}'.format(bbox_position_x,bbox_position_y))
                      print('Speed x {}, speed y {}'.format(bbox_speed_x,bbox_speed_y))
                        
                        
                      break
              # A voir si on garde ou non en fontion des unmatched tracks id
              if not is_poi and False:
                for k in range(len(new_idxs_ut)):
                  if outputs[i][new_idxs_ut[k]][4] == poi_index:
                    is_poi = True
                    
                    bbox = outputs[i][new_idxs_ut[k]][:4] # Bbox from ds
                    conf = outputs[i][new_idxs_ut[k]][4]
                    clss = outputs[i][new_idxs_ut[k]][5]
                    break
                #uses the ds output bbox
                #if names[int(output[5])] == 'person' and output[4] == poi_index and False:
                #  is_poi = True
                #  poi_index_yolo = num # Poi index in the yolo list
                #  bbox = output[:4] # Bbox from yolo
                #  conf = output[4]
                #  clss = output[5]
                  
                  # conf = det[poi_index_yolo,4]
                  # clss = det[poi_index_yolo,5]
          t1_ds = time_synchronized()
          ds_fps.append(1/(t1_ds - t0_ds))
          # End of DeepSort tracking ###########################################
        if len(ds_fps) > 0:
          total_fps_no_print.append(1/(t1_ds - t0_yolo))
        if poi_detected:
          t0_print = time_synchronized()
          
          # Print and show stuff ###############################################
          if is_poi and is_detection:
            if len(outputs[i]) > 0:      
              if colab_webcam and (show_webcam == 'True'):
                  nb_detect_yolo = len(det)
                  nb_detect_ds = len(outputs[i]) 
                  label = '%s %.2f ds id: %s yolor id: %s ds %i yo %i' % (names[int(clss)], conf, poi_index, poi_index_yolo, nb_detect_ds, nb_detect_yolo)
                  plot_one_box(bbox, drawing_array, label=label, color=colors[int(clss)])
                  
                  drawing_array[:,:,0:3] = drawing_array[:,:,0:3]
                  drawing_array[:,:,3] = (drawing_array.max(axis = 2) > 0 ).astype(int) * 255     # Make tranparent background
              
              if colab_webcam and (show_webcam == 'True'):
                # Send bounding box bytes back to javascript
                # print("in")
                drawing_PIL = PIL_Image.fromarray(drawing_array, 'RGBA')
                iobuf = io.BytesIO()
                drawing_PIL.save(iobuf, format='png')
                drawing_bytes = 'data:image/png;base64,{}'.format((str(base64.b64encode(iobuf.getvalue()), 'utf-8')))
                vid_cap.stream_data = drawing_bytes
              else:
                #Print time (inference + NMS)
                print('%f person detected %.3f FPS' % (number_of_person, 1/(t2 - t1)))
                pass
          else:
            if colab_webcam and (show_webcam == 'True'):
              drawing_array = np.zeros([imgsz,imgsz,4], dtype=np.uint8)
              drawing_PIL = PIL_Image.fromarray(drawing_array, 'RGBA')
              iobuf = io.BytesIO()
              drawing_PIL.save(iobuf, format='png')
              drawing_bytes = 'data:image/png;base64,{}'.format((str(base64.b64encode(iobuf.getvalue()), 'utf-8')))
              vid_cap.stream_data = drawing_bytes
          t1_print = time_synchronized()
          print_fps.append(1/(t1_print-t0_print))
          total_fps_print.append(1/(t1_print - t0_yolo))
            
            
    print('Done. (%.3fs)' % (time.time() - t0))
    
    m_yolo_fps = np.mean(yolo_fps)
    m_pose_fps = np.mean(pose_fps)
    m_ds_fps = np.mean(ds_fps)
    m_print_fps = np.mean(print_fps)
    print("overall fps: {:}".format(total_fps_no_print))
    print("Mean FPS:\n YOLO FPS: {:.2f}, Pose FPS: {:.2f}, DS FPS: {:.2f}, Print FPS: {:.2f}".format(m_yolo_fps, m_pose_fps, m_ds_fps, m_print_fps))
    
    
    
