from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os
import time


# .mp4 file to be analyze
vidFile = 'E:\jmuSpring2023\IT445\newTestSpring_2023-02-14-10-21-00__passenger_cam_passenger_left.mp4'

# List to store analyzed frames
analyzedFramesList = []

# number of frames between emotion detection tests
frameInterval = 500

# load video file
vid = cv2.VideoCapture(vidFile)
vid.open(vidFile)

# Loop through frames of video and analyze by the set frameInterval
while vid.isOpen():
    ret, frame = vid.read()

    if ret:
        if vid.get(1) % frameInterval == 0:
            analyzedFrame = frame[:, :, ::-1]

            # Analyze the frame
            result = DeepFace.analyze(analyzedFrame, actions=['emotion'])

            analyzedFramesList.append(result)

    vid.release()

print(analyzedFramesList)
