from deepface import DeepFace
import cv2
import datetime
import sys

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Starts to write to a mp4 file so the user can view later
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('E:/jmuSpring2023/IT445/VideoDetectionOutput/output.mp4', fourcc, 20.0, (640, 480))
startTime = datetime.datetime.now()

# Starts to write the output from the console to a text file
with open('E:/jmuSpring2023/IT445/VideoDetectionOutput/output.txt', 'w') as f:
    while True:
        try:
            # Captures the frames from webcam
            ret, frame = cap.read()

            # Starts timestamp at beginning of recording live video.
            if ret and startTime is None:
                startTime = datetime.datetime.now()

            # Converts the frames to greyscale and equalizes to improve accuracy
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalizedFrame = cv2.equalizeHist(gray)
            processedFrame = cv2.cvtColor(equalizedFrame, cv2.COLOR_GRAY2BGR)

            # Detects the faces from greyscale frames
            faces = face_cascade.detectMultiScale(processedFrame, 1.1, 4)

            # Draws a box around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Crops the detected face and analyzes  emotions
                detected_face = frame[y:y + h, x:x + w]
                emotions = DeepFace.analyze(detected_face, actions=['emotion'])

                # Displays the emotion detection results with the dominant emotion on frame
                for emotion, value in emotions['emotion'].items():
                    dominant_emotion = max(emotions['emotion'], key=emotions['emotion'].get)
                    cv2.putText(frame, f"{dominant_emotion}: {emotions['emotion'][dominant_emotion]:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                    # Gets a timestamp to start from beginning of video
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    elapsed_time = datetime.datetime.now() - startTime
                    timestamp = str(elapsed_time)

                    # Prints the dominant emotion in the python terminal and
                    # Write output to file
                    f.write(f"Timestamp of video: {timestamp} Dominant emotion: {dominant_emotion}\n")
                    print(f"Timestamp of video: {timestamp} Dominant emotion: {dominant_emotion}")

                out.write(frame)
        # Prints an error message if a face couldn't be analyzed
        except ValueError:
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            elapsed_time = datetime.datetime.now() - startTime
            timestamp = str(elapsed_time)
            f.write("Couldn't read face")
            print(f"Timestamp of video: {timestamp}: Couldn't read face")
        # Display the frame for user to view in real time
        cv2.imshow('Emotion Analysis', frame)

        # If "q" is pressed, then the script will end
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release VideoWriter object and close output file
out.release()
f.close()

# Closes the live feed
cap.release()
cv2.destroyAllWindows()
