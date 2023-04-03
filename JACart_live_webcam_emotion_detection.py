from deepface import DeepFace
import cv2
import datetime
import sys

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    try:
        # Captures the frames from webcam
        ret, frame = cap.read()

        # Converts the frames to greyscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detects the faces from greyscale frames
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

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
    # Prints an error message if a face couldn't be analyzed
    except ValueError:
        print("Couldn't read face")
    # Display the frame for user to view in real time
    cv2.imshow('Emotion Analysis', frame)

    # If "q" is pressed, then the script will end
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Closes the live feed
cap.release()
cv2.destroyAllWindows()
