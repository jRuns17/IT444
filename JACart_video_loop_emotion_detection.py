from deepface import DeepFace
import cv2
import datetime

# List to store analyzed frames
analyzedFramesList = []

# number of frames between emotion detection tests
frameInterval = 225

# load video file - short
# vidFile = cv2.VideoCapture('E:/jmuSpring2023/IT445/newTestSpring_2023-02-14-10-21-00__passenger_cam_passenger_left.mp4')

# load video file - long
vidFile = cv2.VideoCapture('E:/jmuFall2022/Capstone/bahaa_2022-11-17-14-21-26__passenger_cam_passenger_left.mp4')
if (vidFile.isOpened() == False):
    print("error opening the video file")
else:
    fps = int(vidFile.get(cv2.CAP_PROP_FPS))
    print("Frame Rate : ", fps, "frames per second")

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

            # Use the following line when you just want the dominant emotion
            dominantEmotion = max(result['emotion'].items(), key=lambda x: x[1])[0]

            timestamp_ms = vidFile.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = str(datetime.timedelta(milliseconds=timestamp_ms))

            # Use the following line when you want all the emotion values
            # analyzedFramesList.append(f"Timestamp of video: {timestamp}  Result: {result}")

            analyzedFramesList.append(f"Timestamp of video: {timestamp}  Result: {dominantEmotion}")
    except ValueError:
        timestamp_ms = vidFile.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = str(datetime.timedelta(milliseconds=timestamp_ms))
        analyzedFramesList.append(f"Timestamp of video: {timestamp}: Image could not be read")
print("Reached end of video")
vidFile.release()

print(analyzedFramesList)

with open(r'E:/jmuSpring2023/IT445/VideoDetectionOutput/test1_Long_Video_Dominant.txt', 'w') as fp:
    for entry in analyzedFramesList:
        fp.write("%s\n" % entry)
