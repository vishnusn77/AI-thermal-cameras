import cv2
import torch
import warnings

# Suppress FutureWarnings related to torch.cuda.amp.autocast
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_people(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_people_count = 0  # To store total count of people across all frames

    while True:
        ret, frame = cap.read()

        # Exit the loop if video has ended or there's a problem with frame capture
        if not ret:
            print(f"End of video. Total people counted: {total_people_count}")
            break

        # Use YOLO model to detect objects in the frame
        results = model(frame)

        # Get the bounding boxes for detected people
        detected_objects = results.pandas().xyxy[0]
        people = detected_objects[detected_objects['name'] == 'person']

        # Count the number of people in the current frame
        people_count = len(people)
        total_people_count += people_count

        # Display the count for the current frame
        print(f"People detected in this frame: {people_count}")

        # Draw bounding boxes around detected people
        for _, person in people.iterrows():
            x1, y1, x2, y2 = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Show the frame with detections
        cv2.imshow('People Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Exiting early. Total people counted: {total_people_count}")
            break

    # Properly release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Test the function on a sample video or webcam
video_path = './sample/footage_1.mp4'  # Replace with your video file path, or use 0 for webcam
detect_people(video_path)
