import cv2
import mediapipe as mp
# import dependencies
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode
import cv2
import PIL
import io
import html
import time
from google.colab.patches import cv2_imshow

import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def pose_detection(frame):
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
      # Recolor image to RGB
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
    
      # Make detection
      results = pose.process(image)

      return results

def pose_analysis(frame):
  results = pose_detection(frame)
  pose_array = np.zeros([640,640,3], dtype=np.uint8)
  pose_array.flags.writeable = True
  pose_array = cv2.cvtColor(pose_array, cv2.COLOR_RGB2BGR)
  vis_threshold = 0.4
  up = False
  
  if results is not None:  
    if results.pose_landmarks is not None:
      # Extract landmarks
      landmarks = results.pose_landmarks.landmark

      # shoulder, elbow, wrist
      left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

      left_vis = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > vis_threshold,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > vis_threshold,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > vis_threshold]

      right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

      right_vis = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > vis_threshold,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > vis_threshold,
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > vis_threshold]
      if (all(right_vis) & (right[2] < right[0]) & (right[2] < right[1])) and (all(left_vis) & (left[2] < left[0]) & (left[2] < left[1])):
        up = True
        mp_drawing.draw_landmarks(pose_array, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    return pose_array, up


def get_mask_for_person(bbox_array, left, top, right, bottom, label):
  person_detected = False #save mask only if a person is detected
  mask = np.zeros([bbox_array.shape[0],bbox_array.shape[1]])
  if label == 'person':
    person_detected = True
    mask[top:bottom, left:right] = 1

  return person_detected, mask

