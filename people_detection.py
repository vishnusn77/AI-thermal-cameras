import cv2
import torch
import numpy as np
from sort import Sort  # SORT tracker

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the SORT tracker
tracker = Sort()

def detect_and_track_people(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    unique_ids = set()  # Set to store unique IDs of tracked people

    while True:
        ret, frame = cap.read()

        # Exit the loop if the video has ended
        if not ret:
            print(f"Total people detected: {len(unique_ids)}")
            break

        # Use YOLOv5 to detect objects in the frame
        results = model(frame)
        detected_objects = results.pandas().xyxy[0]
        people = detected_objects[detected_objects['name'] == 'person']

        # Prepare detections for the tracker: format [xmin, ymin, xmax, ymax, score]
        detections = []
        for _, person in people.iterrows():
            xmin, ymin, xmax, ymax, score = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax']), person['confidence']
            detections.append([xmin, ymin, xmax, ymax, score])

        # Convert to numpy array
        detections = np.array(detections)

        # Update the tracker with current detections
        tracked_objects = tracker.update(detections)

        # Loop through tracked objects
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])

            # If the track_id is new, add it to the unique_ids set
            if track_id not in unique_ids:
                unique_ids.add(track_id)

            # Draw the bounding boxes and track ID on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Person: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Show the frame with the tracking boxes
        cv2.imshow('People Detection and Tracking', frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Exiting early. Total people detected: {len(unique_ids)}")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Test the function on a sample video
video_path = './sample/footage_4.mp4'  # Replace with your video file path, or use 0 for webcam
detect_and_track_people(video_path)
