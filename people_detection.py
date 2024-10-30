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

# Define crowd threshold for alerts
CROWD_THRESHOLD = 9  # Set this to the desired maximum crowd size

def detect_and_track_people(video_source="webcam", video_path=None, rtsp_url=None):
    if video_source == "webcam":
        cap = cv2.VideoCapture(0)
    elif video_source == "video" and video_path:
        cap = cv2.VideoCapture(video_path)
    elif video_source == "rtsp" and rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
    else:
        print("Error: Invalid video source or missing parameters.")
        return

    if video_source == "rtsp":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    unique_ids = set()  # Set to store unique IDs of tracked people

    while True:
        ret, frame = cap.read()
        
        # If video ends, print final crowd count without error message
        if not ret:
            print(f"Crowd limit is: {CROWD_THRESHOLD}")
            print(f"Total people detected: {len(unique_ids)}")
            break

        # Resize frame for RTSP to speed up YOLO inference
        resized_frame = cv2.resize(frame, (320, 240)) if video_source == "rtsp" else frame

        # Detect people using YOLOv5
        results = model(resized_frame)
        detected_objects = results.pandas().xyxy[0]
        people = detected_objects[detected_objects['name'] == 'person']

        # Prepare detections for tracker
        detections = []
        for _, person in people.iterrows():
            if video_source == "rtsp":
                xmin, ymin, xmax, ymax, score = int(person['xmin'] * 2), int(person['ymin'] * 2), int(person['xmax'] * 2), int(person['ymax'] * 2), person['confidence']
            else:
                xmin, ymin, xmax, ymax, score = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax']), person['confidence']
            detections.append([xmin, ymin, xmax, ymax, score])

        # Convert to numpy array
        detections = np.array(detections)

        # Update the tracker
        tracked_objects = tracker.update(detections) if len(detections) > 0 else []

        # Draw bounding boxes and track ID on the frame
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            if track_id not in unique_ids:
                unique_ids.add(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Person: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Display crowd count and alert message if crowd exceeds threshold
        total_people = len(unique_ids)
        total_people_text = f'Total People: {total_people}'
        cv2.putText(frame, total_people_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Check if crowd count exceeds the threshold
        if total_people > CROWD_THRESHOLD:
            alert_text = "ALERT: Crowd limit exceeded!"
            cv2.putText(frame, alert_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            print(f"Crowd limit is: {CROWD_THRESHOLD}")
            print(alert_text)

        # Show the frame with bounding boxes and alerts
        cv2.imshow('People Detection and Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Crowd limit is: {CROWD_THRESHOLD}")
            print(f"Total people detected: {total_people}")
            break

    cap.release()
    cv2.destroyAllWindows()
