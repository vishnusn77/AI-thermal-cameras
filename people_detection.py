import cv2
import torch
import numpy as np
import warnings
import time
from sort import Sort  # SORT tracker
import pytesseract
import re
from statistics import mode

# Set up Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the SORT tracker with updated parameters for better tracking persistence
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.25)

# Define crowd threshold for alerts
CROWD_THRESHOLD = 10  # Set this to the desired maximum crowd size
SCREENSHOT_INTERVAL = 0.25  # Capture screenshots every 0.5 seconds

# List to store detected temperatures
temperature_readings = []

# Function to extract green text from the image for temperature reading
def extract_green_text(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([55, 100, 60])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    green_text = cv2.bitwise_and(image, image, mask=mask)
    gray_text = cv2.cvtColor(green_text, cv2.COLOR_BGR2GRAY)
    _, thresholded_text = cv2.threshold(gray_text, 120, 255, cv2.THRESH_BINARY)
    blurred_text = cv2.GaussianBlur(thresholded_text, (5, 5), 0)
    
    text = pytesseract.image_to_string(blurred_text, config='--psm 6')
    cleaned_text = re.sub(r'[^0-9.]', '', text)
    if cleaned_text:
        try:
            return float(cleaned_text)
        except ValueError:
            return None
    return None

# Function to capture and process screenshot without saving
def capture_screenshot(frame):
    temperature = extract_green_text(frame)
    if temperature is not None:
        temperature_readings.append(temperature)  # Store detected temperature
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

    if video_source == "rtsp":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    unique_ids = set()
    last_screenshot_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Crowd limit is: {CROWD_THRESHOLD}")
            print(f"Total people detected: {len(unique_ids)}")
            break

        resized_frame = cv2.resize(frame, (320, 240)) if video_source == "rtsp" else frame
        results = model(resized_frame)
        detected_objects = results.pandas().xyxy[0]
        people = detected_objects[(detected_objects['name'] == 'person') & (detected_objects['confidence'] >= 0.5)]

        detections = []
        for _, person in people.iterrows():
            if video_source == "rtsp":
                xmin, ymin, xmax, ymax, score = int(person['xmin'] * 2), int(person['ymin'] * 2), int(person['xmax'] * 2), int(person['ymax'] * 2), person['confidence']
            else:
                xmin, ymin, xmax, ymax, score = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax']), person['confidence']
            detections.append([xmin, ymin, xmax, ymax, score])

        detections = np.array(detections)
        tracked_objects = tracker.update(detections) if len(detections) > 0 else []

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            if track_id not in unique_ids:
                unique_ids.add(track_id)
            
            color = (173, 216, 230)  # Light Blue
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Person: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        current_time = time.time()
        if video_source == "rtsp" and current_time - last_screenshot_time >= SCREENSHOT_INTERVAL:
            temperature = capture_screenshot(frame)  # Store temperature readings
            last_screenshot_time = current_time
            if temperature is not None:
                print(f"Detecting: {temperature}")

        total_people = len(unique_ids)
        cv2.putText(frame, f'Total People: {total_people}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if total_people > CROWD_THRESHOLD:
            cv2.putText(frame, "ALERT: Crowd limit exceeded!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow('People Detection and Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print the most frequent temperature detected during the stream
    if temperature_readings:
        try:
            final_temperature = mode(temperature_readings)
            print(f"Temperature detected: {final_temperature}°C")
        except:
            print("No consistent temperature reading available.")
    else:
        print("No temperature readings captured.")
