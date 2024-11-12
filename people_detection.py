import cv2
import torch
import numpy as np
import warnings
import time
from sort import Sort  # SORT tracker
import pytesseract
import re

# Set up Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the SORT tracker with updated parameters for better tracking persistence
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.25)

# Define crowd threshold for alerts
CROWD_THRESHOLD = 0  # Set this to the desired maximum crowd size
SCREENSHOT_INTERVAL = 0.25  # Capture screenshots every 0.5 seconds

# Function to extract green text from the image for temperature reading
def extract_green_text(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for green color
    lower_green = np.array([55, 100, 60])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Extract green areas
    green_text = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to grayscale for OCR processing
    gray_text = cv2.cvtColor(green_text, cv2.COLOR_BGR2GRAY)
    
    # Preprocess the image to improve OCR results
    _, thresholded_text = cv2.threshold(gray_text, 120, 255, cv2.THRESH_BINARY)
    blurred_text = cv2.GaussianBlur(thresholded_text, (5, 5), 0)
    
    # Use OCR to extract text
    text = pytesseract.image_to_string(blurred_text, config='--psm 6')
    
    # Clean up the extracted text to isolate numeric values
    cleaned_text = re.sub(r'[^0-9.]', '', text)
    if cleaned_text:
        try:
            temperature = float(cleaned_text)
            return temperature
        except ValueError:
            return "Invalid Temperature"
    else:
        return "No Temperature Found"

# Function to capture and process screenshot without saving
def capture_screenshot(frame):
    # Extract temperature data from the green text in the frame
    temperature = extract_green_text(frame)
    return temperature

# Main function for detection, tracking, and monitoring
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

    # Set RTSP-specific parameters for buffering
    if video_source == "rtsp":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    unique_ids = set()  # Set to store unique IDs of tracked people
    start_time = time.time()
    last_screenshot_time = 0
    last_temperature = None  # Variable to store the last captured temperature

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Crowd limit is: {CROWD_THRESHOLD}")
            print(f"Total people detected: {len(unique_ids)}")
            break

        # Resize frame for speed optimization
        resized_frame = cv2.resize(frame, (320, 240)) if video_source == "rtsp" else frame

        # YOLOv5 detection
        results = model(resized_frame)
        detected_objects = results.pandas().xyxy[0]
        people = detected_objects[(detected_objects['name'] == 'person') & (detected_objects['confidence'] >= 0.5)]

        # Prepare detections for the tracker
        detections = []
        for _, person in people.iterrows():
            if video_source == "rtsp":
                xmin, ymin, xmax, ymax, score = int(person['xmin'] * 2), int(person['ymin'] * 2), int(person['xmax'] * 2), int(person['ymax'] * 2), person['confidence']
            else:
                xmin, ymin, xmax, ymax, score = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax']), person['confidence']
            detections.append([xmin, ymin, xmax, ymax, score])

        detections = np.array(detections)

        # Update tracker
        tracked_objects = tracker.update(detections) if len(detections) > 0 else []

        # Draw bounding boxes and track ID on frame
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            if track_id not in unique_ids:
                unique_ids.add(track_id)
            
            # Set a more subtle color (light blue)
            color = (173, 216, 230)  # Light Blue (subtle)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Person: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Capture screenshot at intervals and extract temperature
        current_time = time.time()
        if current_time - last_screenshot_time >= SCREENSHOT_INTERVAL:
            last_temperature = capture_screenshot(frame)  # Store last captured temperature
            last_screenshot_time = current_time
            print(f"Temperature: {last_temperature}")

        # Crowd count and alert display
        total_people = len(unique_ids)
        cv2.putText(frame, f'Total People: {total_people}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        if total_people > CROWD_THRESHOLD:
            alert_text = f"⚠️ ALERT: Crowd limit exceeded!\nTemperature: {last_temperature}\nTotal People: {total_people}"
            cv2.putText(frame, "ALERT: Crowd limit exceeded!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            alert_text = f"Temperature: {last_temperature}\nTotal People: {total_people}"

        # Display the frame
        cv2.imshow('People Detection and Tracking', frame)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()