import cv2
from deepface import DeepFace
from mtcnn import MTCNN

# Load the video file
cap = cv2.VideoCapture('E:/jmuSpring2023/IT445/baselineTestShort_2023-03-28-17-29-33__passenger_cam_passenger_left.mp4')

# Get the video codec and frame rate
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Get the size of the frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to write the output video
out = cv2.VideoWriter('E:/jmuSpring2023/IT445/VideoDetectionOutput/old-baseline-overlay-4-26.mp4', fourcc, fps, (frame_width, frame_height))

# Create an MTCNN face detector
detector = MTCNN()

# Loop over the frames of the video
while cap.isOpened():
    try:
        # Read the frame from the video
        ret, frame = cap.read()

        # If the frame cannot be read, break the loop
        if not ret:
            break

        # Detect faces in the frame
        result = detector.detect_faces(frame)

        # Process each detected face
        for face in result:
            # Extract the bounding box coordinates of the face
            x, y, w, h = face['box']

            # Draw a bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Analyze the emotions in the face
            detected_face = frame[y:y+h, x:x+w]
            emotions = DeepFace.analyze(detected_face, actions=['emotion'])

            # Determine the dominant emotion
            dominant_emotion = max(emotions['emotion'], key=emotions['emotion'].get)

            # Draw the emotion label above the bounding box
            cv2.putText(frame, f"{dominant_emotion}: {emotions['emotion'][dominant_emotion]:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    except ValueError:
        print("Couldnt read face")
    # Write the frame to the output video
    out.write(frame)

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()
