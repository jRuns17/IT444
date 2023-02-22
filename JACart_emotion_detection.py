from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os


# Load and display image
img = cv2.imread('E:\jmuSpring2023\IT445\cartPassenger.PNG')
# img = cv2.imread('E:\jmuSpring2023\IT445\myHappyEmotion.jpg')
plt.imshow(img[:,:,::-1])
plt.show()

# Analyze and print results
print("0")
result = DeepFace.analyze(img, actions = ['emotion'])
print("1")
print(result)
print("2")
