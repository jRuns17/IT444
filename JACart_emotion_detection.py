from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os


# Load and display image
img = cv2.imread('E:\jmuSpring2023\445 presentation 1\cartPassenger.PNG')
# img = cv2.imread('E:\jmuSpring2023\IT445\myHappyEmotion.jpg')


# Analyze and print results
print("0")
result = DeepFace.analyze(img, actions = ['emotion'])
print("1")
print(result)
print("2")
