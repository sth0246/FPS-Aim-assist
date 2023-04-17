# -*- coding: utf-8 -*-
# @Time    : 2022/6/23 14:20
# @Author  : sth0246
# @File    : mytestpose.py
# @Description : 使用ImageGrab获取屏幕图像，MediaPipe识别人体架构，然后pydirectinput移动鼠标至头部
import time

import cv2
import mediapipe as mp
from PIL import ImageGrab
import numpy as np
import pydirectinput as pyautogui
print(pyautogui.size())
width,height=pyautogui.size()
count = 0
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
left = 700
top = 330
right = 1220
bottom = 750
x = 0
y = 0
# For webcam input:
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while 1:
    # success, image = cap.read()
    # if not success:
    #   print("Ignoring empty camera frame.")
    #   # If loading a video, use 'break' instead of 'continue'.
    #   continue

    img_np = ImageGrab.grab(bbox=(left, top, right, bottom))
    image = np.array(img_np)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    if results.pose_landmarks:

        x = results.pose_landmarks.landmark[0].x*(right-left)
        y = results.pose_landmarks.landmark[0].y*(bottom-top)
        count += 1
        print(count)
        pyautogui.moveTo(int(x+left), int(y+top))
        # mouse_x,mouse_y = pyautogui.position()
        # print('real position',pyautogui.position())
        # pyautogui.moveRel(int(x+left-mouse_x),int(y+top-mouse_y))
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break