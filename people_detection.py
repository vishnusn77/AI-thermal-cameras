import cv2
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sort import Sort  # SORT tracker

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Initialize the SORT tracker with updated parameters for better tracking persistence
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.25)

def detect_and_track_people(video_source="webcam", video_path=None, rtsp_url=None):
    """
    Detect and track people from different video sources.

    Parameters:
    - video_source: 'webcam' | 'video' | 'rtsp'
    - video_path: If video_source is 'video', provide the video file path.
    - rtsp_url: If video_source is 'rtsp', provide the RTSP stream URL.
    """
    
    if video_source == "webcam":
        # Open live webcam stream (default webcam)
        cap = cv2.VideoCapture(0)
    elif video_source == "video" and video_path:
        # Open video file
        cap = cv2.VideoCapture(video_path)
    elif video_source == "rtsp" and rtsp_url:
        # Connect to the Provix camera using the RTSP URL
        cap = cv2.VideoCapture(rtsp_url)
    else:
        print("Error: Invalid video source or missing parameters.")
        return

    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Optionally, set frame size and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    unique_ids = set()  # Set to store unique IDs of tracked people

    while True:
        ret, frame = cap.read()

        # Exit the loop if there are no frames captured
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Resize frame to speed up YOLO inference
        resized_frame = cv2.resize(frame, (320, 240))

        # Use YOLOv5 to detect objects in the resized frame
        results = model(resized_frame)

        # Filter out low-confidence detections to speed up processing
        detected_objects = results.pandas().xyxy[0]
        people = detected_objects[(detected_objects['name'] == 'person') & (detected_objects['confidence'] > 0.4)]

        # Prepare detections for the tracker: format [xmin, ymin, xmax, ymax, score]
        detections = []
        for _, person in people.iterrows():
            xmin, ymin, xmax, ymax, score = int(person['xmin'] * 2), int(person['ymin'] * 2), int(person['xmax'] * 2), int(person['ymax'] * 2), person['confidence']
            detections.append([xmin, ymin, xmax, ymax, score])

        # Convert to numpy array
        detections = np.array(detections)

        # Update the tracker with current detections, only if there are detections
        if len(detections) > 0:
            tracked_objects = tracker.update(detections)
        else:
            tracked_objects = []

        # Loop through tracked objects and draw them on the frame
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])

            # If the track_id is new, add it to the unique_ids set
            if track_id not in unique_ids:
                unique_ids.add(track_id)

            # Draw the bounding boxes and track ID on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Bounding box in blue (BGR: 255, 0, 0)
            cv2.putText(frame, f'Person: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # Text in white

        # Display the total number of unique people detected
        total_people_text = f'Total People: {len(unique_ids)}'
        cv2.putText(frame, total_people_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # Text in red

        # Show the frame with the tracking boxes and total count
        cv2.imshow('People Detection and Tracking', frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Exiting early. Total people detected: {len(unique_ids)}")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
