from deepface import DeepFace
import cv2
import datetime
import sys
import pyzed.sl as sl

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set the camera configuration
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return

    # Create a video recorder object
    video_params = sl.RecordingParameters("output.mp4", sl.SVO_COMPRESSION_MODE.H264)
    video_recorder = sl.Recorder()
    err = video_recorder.open(video_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to create video recorder")
        zed.close()
        return

    # Capture and record frames
    runtime_params = sl.RuntimeParameters()
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image from the ZED camera
            left_image = sl.Mat()
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            # Record the left image
            err = video_recorder.record(left_image)
            if err != sl.ERROR_CODE.SUCCESS:
                print("Failed to record frame")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the camera and video recorder
    zed.close()
    video_recorder.close()

if __name__ == "__main__":
    main()
