from ultralytics import YOLO
import cv2
import numpy as np
import time
import argparse
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", action="store_true")
    args = parser.parse_args()

    model = YOLO("./Models/gesture_tracker_v0.pt")

    video_path = "./Video/istockphoto-1920129337-640_adpp_is.mp4"
    
    if args.r:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    T = 1 / fps
    curr_time = 0
    trajectory = []

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Extract keypoint data
            keypoints = results[0].keypoints.xyn.cpu().numpy()

            curr_time += T
            if len(keypoints) >= 2:
                trajectory.append([np.array(keypoints[0]), np.array(keypoints[1]), curr_time])
            elif len(keypoints) == 1:
                trajectory.append([np.array(keypoints[0]), np.zeros((20,2), np.float32), curr_time])

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    timestamp = time.time()
    value = datetime.datetime.fromtimestamp(timestamp)

    path = f"Trajectory/trajectory_{value.strftime('%Y-%m-%d_%H-%M-%S')}.npy"
    np.save(path, np.array(trajectory, dtype=object), allow_pickle=True)