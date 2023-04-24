##### Load libraries -----
 
import cv2
import mediapipe as mp
import numpy as np
import math
import os


##### Set human pose function -----

def human_pose_features(pose_est, cap):

  landmarks = []

  # full clip
  for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
      _, img = cap.read()
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      # process the frame for pose landmarks
      pose_results = pose_est.process(img)
      keypoints = []
      if pose_results.pose_landmarks is None:
        keypoints = [0.0] * 99
      else:
        for data_point in pose_results.pose_landmarks.landmark:
            keypoints.append(data_point.x)
            keypoints.append(data_point.y)
            keypoints.append(data_point.z)

      landmarks.append(keypoints)

  cap.release()

  landmarks = np.array(landmarks)

  return(landmarks)
  
  
##### Video Human Pose Feature Extraction -----

if __name__ == '__main__':
    
    # code edited from source: https://google.github.io/mediapipe/solutions/pose.html
    
    ## initialize pose estimator
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1, smooth_landmarks=True)
    
    
    ##### Training Videos -----
    
    # input directory of raw videos
    files = sorted(os.listdir('data/videos_160'))
    
    # run background removal on all videos
    for i in files:
      print(i)
      # read input video
      cap = cv2.VideoCapture('data/videos_160/{}'.format(i))
      
      # extract frame keypoints
      landmarks = human_pose_features(pose, cap)

      # save numpy array
      np.save('data/videos_160_human_pose/{}.npy'.format(i.split('.mp4')[0]), landmarks)