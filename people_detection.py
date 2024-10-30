import cv2
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sort import Sort  # SORT tracker

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the SORT tracker with updated parameters for better tracking persistence
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.25)

def detect_and_track_people(video_source="webcam", video_path=None, rtsp_url=None):
    # Set up video capture based on source
    if video_source == "webcam":
        cap = cv2.VideoCapture(0)
    elif video_source == "video" and video_path:
        cap = cv2.VideoCapture(video_path)
    elif video_source == "rtsp" and rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
    else:
        print("Error: Invalid video source or missing parameters.")
        return

    # Apply RTSP-specific optimizations
    if video_source == "rtsp":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for RTSP to lower latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to lighten load
    else:
        # Default settings for webcam and video file
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    unique_ids = set()  # Set to store unique IDs of tracked people

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Apply frame resizing only for RTSP stream to improve processing speed
        if video_source == "rtsp":
            resized_frame = cv2.resize(frame, (320, 240))
        else:
            resized_frame = frame  # Keep original resolution for webcam and video file

        # Use YOLOv5 to detect objects in the chosen frame size
        results = model(resized_frame)

        # Filter detections with higher confidence threshold
        detected_objects = results.pandas().xyxy[0]
        people = detected_objects[(detected_objects['name'] == 'person') & (detected_objects['confidence'] > 0.5)]

        # Prepare detections for the tracker: [xmin, ymin, xmax, ymax, score]
        detections = []
        for _, person in people.iterrows():
            if video_source == "rtsp":
                # Scale bounding boxes back up after resizing
                xmin, ymin, xmax, ymax, score = int(person['xmin'] * 2), int(person['ymin'] * 2), int(person['xmax'] * 2), int(person['ymax'] * 2), person['confidence']
            else:
                # Use original bounding box coordinates for other sources
                xmin, ymin, xmax, ymax, score = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax']), person['confidence']
            detections.append([xmin, ymin, xmax, ymax, score])

        detections = np.array(detections)

        # Update the tracker with current detections if there are any
        tracked_objects = tracker.update(detections) if len(detections) > 0 else []

        # Draw bounding boxes and display track IDs
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])

            if track_id not in unique_ids:
                unique_ids.add(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Person: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Display total number of unique people detected
        total_people_text = f'Total People: {len(unique_ids)}'
        cv2.putText(frame, total_people_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Show the frame with bounding boxes and total count
        cv2.imshow('People Detection and Tracking', frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Exiting early. Total people detected: {len(unique_ids)}")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
