from ultralytics import YOLO
import cv2
import numpy as np
import time
import argparse
import datetime


def video_inference(model, video_path, args):
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
                trajectory.append([np.array(keypoints[0]), np.zeros((21,2), np.float32), curr_time])

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


def image_inference(model, image_path):
    img = cv2.imread(image_path)

    # Run YOLO inference on the frame
    results = model(img)

    # Visualize the results on the frame
    annotated_image = results[0].plot()

    # Extract keypoint data
    keypoints = results[0].keypoints.xyn.cpu().numpy()

    # Print keypoints
    if len(keypoints) >= 2:
        print(f"Hand 1: {keypoints[0]}")
        print(f"Hand 2: {keypoints[1]}")
        kp = [np.array(keypoints[0]), np.array(keypoints[1])]
    elif len(keypoints) == 1:
        print(f"Hand 1: {keypoints[0]}")
        kp = [np.array(keypoints[0]), np.zeros((21,2), np.float32)]

    timestamp = time.time()
    value = datetime.datetime.fromtimestamp(timestamp)

    path = f"Trajectory/keypoints_{value.strftime('%Y-%m-%d_%H-%M-%S')}.npy"
    np.save(path, np.array(kp, dtype=object), allow_pickle=True)

    # Display the annotated image
    cv2.imshow("YOLO Inference", annotated_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", action="store_true")
    parser.add_argument("-i", action="store_true")
    args = parser.parse_args()

    model = YOLO("./Models/gesture_tracker_v2.pt")

    video_path = "./Video/istockphoto-1920129337-640_adpp_is.mp4"
    image_path = "./Img/istockphoto-878155798-612x612.jpg"

    if args.i:
        image_inference(model, image_path)
    else:
        video_inference(model, video_path, args)