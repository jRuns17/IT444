from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os
import time

# .mp4 file to be analyze
#vidFile = 'E:\jmuSpring2023\IT445\newTestSpring_2023-02-14-10-21-00__passenger_cam_passenger_left.mp4'

# List to store analyzed frames
analyzedFramesList = []

# number of frames between emotion detection tests
frameInterval = 30

# load video file
vidFile = cv2.VideoCapture('E:/jmuSpring2023/IT445/newTestSpring_2023-02-14-10-21-00__passenger_cam_passenger_left.mp4')

if(vidFile.isOpened() == False):
    print("error opening the video file")
else:
    fps = int(vidFile.get(5))
    print("Frame Rate : ",fps,"frames per second")

# Loop through frames of video and analyze by the set frameInterval
while vidFile.isOpened():
    try:
        ret, frame = vidFile.read()
        if not ret:
            break
        if vidFile.get(1) % frameInterval == 0:
            analyzedFrame = frame[:, :, ::-1]

            # Analyze the frame
            result = DeepFace.analyze(analyzedFrame, actions=['emotion'])

            analyzedFramesList.append(result)
    except ValueError:
        analyzedFramesList.append('Image could not be read')
print("Reached end of video")
vidFile.release()

print(analyzedFramesList)
