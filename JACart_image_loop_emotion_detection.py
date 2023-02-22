from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os
import time
plt.ion()

# Folder where images to be analyzed are stored
imageFolder = 'E:\jmuSpring2023\IT445'


# Loops through the files that end with .jpg and displays them before analyzing
for files in os.listdir(imageFolder):
    if files.endswith('.jpg'):
        # Load and display image
        img = cv2.imread(os.path.join(imageFolder, files))
        #plt.imshow(img[:,:,::-1])
        #plt.show()
        #plt.close()

        # Analyze and print results along with filename
        result = DeepFace.analyze(img, actions = ['emotion'])
        print(files)
        print(result)
        time.sleep(5)

