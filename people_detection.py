import cv2
import torch
import numpy as np
import pytesseract
import re
from sort import Sort  # SORT tracker
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the SORT tracker with updated parameters for better tracking persistence
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.25)

# Define thresholds
CROWD_THRESHOLD = 2  # Maximum number of people before triggering a crowd alert
TEMPERATURE_THRESHOLD = 35.0  # Temperature threshold for alerts
SCREENSHOT_INTERVAL = 0.25  # Capture screenshots every 0.25 seconds

# Dictionary to store the highest temperature for each person
person_temperatures = {}

# Function to extract green text from the image for temperature reading
def extract_green_text(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 60])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_text = cv2.bitwise_and(image, image, mask=mask)
    gray_text = cv2.cvtColor(green_text, cv2.COLOR_BGR2GRAY)
    _, thresholded_text = cv2.threshold(gray_text, 120, 255, cv2.THRESH_BINARY)
    blurred_text = cv2.GaussianBlur(thresholded_text, (5, 5), 0)
    text = pytesseract.image_to_string(blurred_text, config='--psm 6')
    cleaned_text = re.sub(r'[^0-9.]', '', text)
    if cleaned_text:
        try:
            temperature = float(cleaned_text)
            if 30.0 <= temperature <= 42.0:
                return temperature
            else:
                return "Invalid Temperature"
        except ValueError:
            return "Invalid Temperature"
    else:
        return "No Temperature Found"

# Main function for detection, tracking, and monitoring
def detect_and_track_people(video_source="webcam", video_path=None, rtsp_url=None, alert_callback=None):
    if video_source == "webcam":
        cap = cv2.VideoCapture(0)
    elif video_source == "video" and video_path:
        cap = cv2.VideoCapture(video_path)
    elif video_source == "rtsp" and rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
    else:
        print("Error: Invalid video source or missing parameters.")
        return

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
            xmin, ymin, xmax, ymax, score = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax']), person['confidence']
            height_expansion = 70
            ymin_expanded = max(0, ymin - height_expansion)
            ymax_expanded = min(frame.shape[0], ymax + height_expansion)
            detections.append([xmin, ymin_expanded, xmax, ymax_expanded, score])

        detections = np.array(detections)
        tracked_objects = tracker.update(detections) if len(detections) > 0 else []

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            if track_id not in unique_ids:
                unique_ids.add(track_id)

            color = (173, 216, 230)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Person: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        current_time = time.time()
        if current_time - last_screenshot_time >= SCREENSHOT_INTERVAL:
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.shape[0] == 0 or person_roi.shape[1] == 0:
                    print(f"Invalid ROI for Person {track_id}: {x1}, {y1}, {x2}, {y2}")
                    continue

                temperature = extract_green_text(person_roi)
                if temperature != "No Temperature Found" and temperature != "Invalid Temperature":
                    temperature = round(temperature, 1)
                    if track_id in person_temperatures:
                        if temperature > person_temperatures[track_id]:
                            person_temperatures[track_id] = temperature
                    else:
                        person_temperatures[track_id] = temperature

            total_people = len(unique_ids)
            exceeding_temperatures = [
                temp for temp in person_temperatures.values() if temp > TEMPERATURE_THRESHOLD
            ]

            # Crowd Alert
            crowd_status = "Exceeded" if total_people > CROWD_THRESHOLD else "Normal"
            crowd_alert_message = (
                "⚠️ Area overcrowded! Please ensure social distancing."
                if crowd_status == "Exceeded" else ""
            )

            crowd_message = (
                f"- Crowd capacity: {CROWD_THRESHOLD}\n"
                f"- Total people detected: {total_people}\n"
                f"- Status: {crowd_status}\n"
                # f"{crowd_alert_message}"
            )
            crowd_color = "#ff6347" if total_people > CROWD_THRESHOLD else "#32CD32"

            # Temperature Alert
            temp_status = "Exceeded" if len(exceeding_temperatures) >= 2 else "Normal"
            temp_alert_message = (
                "⚠️ Elevated body temperature detected! Please wear a mask."
                if temp_status == "Exceeded" else ""
            )

            temp_message = (
                f"- Temperature threshold: {TEMPERATURE_THRESHOLD}°C\n"
                f"- Detected Temperatures: {', '.join([f'{temp}°C' for temp in person_temperatures.values()])}\n"
                f"- Status: {temp_status}\n"
                # f"{temp_alert_message}"
            )
            temp_color = "#ff4500" if len(exceeding_temperatures) >= 2 else "#32CD32"

            # Send alerts
            if alert_callback:
                alert_callback({
                    "type": "crowd",
                    "message": crowd_message,
                    "color": crowd_color,
                    "alert_message": crowd_alert_message  # Pass the alert message separately
                })
                alert_callback({
                    "type": "temperature",
                    "message": temp_message,
                    "color": temp_color,
                    "alert_message": temp_alert_message  # Pass the alert message separately
                })

            last_screenshot_time = current_time

        cv2.imshow('People Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
