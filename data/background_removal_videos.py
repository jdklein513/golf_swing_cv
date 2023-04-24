 
##### Load libraries -----
 
import cv2
import mediapipe as mp
import numpy as np
import math
import os

def remove_background_videos(pose_est, cap, fourcc, out):
    # run segmentation and write output
      while cap.isOpened():
          # read frame
          _, frame = cap.read()
          try:
              # convert to RGB
              frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              
              # process the frame for pose detection
              pose_results = pose.process(frame_rgb)
              
              # Draw pose segmentation.
              annotated_image = frame.copy()
              red_img = np.zeros_like(annotated_image, dtype=np.uint8)
              red_img[:, :] = (255,255,255)
              if pose_results.segmentation_mask is None:
                segm_2class = 0.1 + 0.8 * np.zeros((dim, dim))
              else:
                segm_2class = 0.1 + 0.8 * pose_results.segmentation_mask
              segm_2class = np.repeat(segm_2class[..., np.newaxis], 3, axis=2)
              annotated_image = annotated_image * segm_2class + red_img * (1 - segm_2class)
              out.write(np.uint8(annotated_image))
    
          except:
              break
              
          if cv2.waitKey(1) == ord('q'):
              break
    
      cap.release()
      out.release()
      cv2.destroyAllWindows()


if __name__ == '__main__':
    
    # code edited from source: https://google.github.io/mediapipe/solutions/pose.html
    
    # initialize pose estimator
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2, smooth_landmarks=True, enable_segmentation=True, smooth_segmentation=True)
    
    
    ##### Training Videos -----
    
    # input directory of raw videos
    files = os.listdir('/data/videos_160')
    
    # set video dimension
    dim = 160
    
    # run background removal on all videos
    for i in files:
      print(i)
      # read input video
      cap = cv2.VideoCapture('/data/videos_160/{}'.format(i))
      
      # set output video format and location
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      out = cv2.VideoWriter('/data/videos_160_segmented/{}'.format(i),
                            fourcc, 
                            cap.get(cv2.CAP_PROP_FPS), 
                            (dim, dim))
      
      remove_background_videos(pose, cap, fourcc, out)
    
    
    ##### Test Videos -----
      
    # read input video
    cap = cv2.VideoCapture('test_video.mp4')
    
    # set output video format and location
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('test_video_segmented.mp4',
                          fourcc, 
                          cap.get(cv2.CAP_PROP_FPS), 
                          (dim, dim))
                          
    remove_background_videos(pose, cap, fourcc, out)
