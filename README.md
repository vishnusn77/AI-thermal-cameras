﻿# CrowdWatch AI: Real-Time Thermal and Crowd Monitoring System

This project is designed to monitor public health and manage crowd density in real-time using advanced thermal cameras, computer vision techniques, and dynamic alert displays. The system extracts temperature data from video feeds using OCR, tracks individual temperatures, and displays dynamic health and safety messages on external monitors.

---

## Features

### 🔥 Real-Time Temperature Detection
- Extracts temperature readings using OCR from thermal camera feeds.

### 🧑‍🤝‍🧑 Dynamic Crowd Monitoring
- Tracks the number of individuals and adjusts occupancy limits in real-time.

### 🚨 Safety Alerts
- Displays warnings for elevated temperatures and overcrowding on external monitors.

### 🔗 Seamless Integration
- Fully compatible with HK Provix cameras and Tesseract OCR.

---

## System Prerequisites

### Hardware Requirements
- **HK Provix Camera**
  - Camera 01: Normal video feed with temperature readings.
  - Camera 02: Thermal feed for enhanced monitoring.
- **External Monitor**: HDMI-supported display for alert notifications.

### Software Requirements
- **Python**: Version 3.8 or higher.
- **Tesseract OCR**: Properly installed and configured (e.g., `C:\Program Files (x86)\Tesseract-OCR` on Windows).

---

## How to Set Up and Run the System

### Step 1: Clone the Repository
```bash
git clone https://github.com/vishnusn77/AI-thermal-cameras.git  
cd AI-thermal-cameras  
```

## Step 2: Set Up Python Environment

Create and activate a virtual environment:

```bash
python -m venv venv  
.\venv\Scripts\activate  # For Windows  
source venv/bin/activate # For macOS/Linux  
```

## Step 3: Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt  
```

## Step 4: Clone YOLOv5 Repository

Download the YOLOv5 model repository into the project folder:

```bash
git clone https://github.com/ultralytics/yolov5  
```
## Step 5: Configure the Cameras

Ensure the following setup:

### Camera 01 RTSP URL:
```bash
rtsp://admin:Admin12345@192.168.1.142/Streaming/channels/2
```
## Step 6: Run the Application

Start the main application:

```bash
python app.py
```

## How It Works

### Temperature Extraction
- Captures frames every 0.25 seconds.
- Uses OpenCV to isolate temperature data displayed in green.
- Extracts text using Tesseract OCR with enhanced preprocessing techniques.

### Crowd Monitoring
- Detects and tracks individuals in the scene using YOLOv5.
- SORT algorithm assigns unique IDs to track individuals across frames.
- Links extracted temperatures with bounding boxes for each person.

### Alert Mechanism
Monitors conditions in real-time:
- **High Temperature Alert**: Triggered if temperature > 37.5°C.
- **Crowd Density Alert**: Triggered if the number of people exceeds the threshold (e.g., 50).
- Displays actionable messages on external monitors using a Tkinter-based GUI.

## File Structure

| File/Folder         | Description                                            |
|---------------------|--------------------------------------------------------|
| `people_detection.py` | Handles people detection, tracking, and temperature extraction. |
| `sort.py`            | Implements the SORT algorithm for unique ID tracking. |
| `app.py`             | Main script integrating all modules for execution.    |
| `alert_display.py`   | Displays alerts dynamically on external monitors.      |
| `requirements.txt`   | List of required Python libraries.                     |
| `yolov5/`            | Repository containing YOLOv5 object detection model.   |


## Customization

### Temperature Thresholds
Update the threshold for temperature alerts in `app.py`:

```python
temperature_threshold = 37.5  
```
## Occupancy Limits
Adjust the maximum allowable crowd size:

```python
max_occupancy = 50  
```
## Troubleshooting

### Common Issues

- **Tesseract OCR Not Found**:  
  Ensure Tesseract is installed, and its path is correctly configured in `app.py`.

- **Camera Not Accessible**:  
  Verify the RTSP URL and ensure cameras are connected to the same network.

- **Dependency Errors**:  
  Run `pip install -r requirements.txt` again in the correct virtual environment.

### Debugging Tips
- Use debug logs or `print()` statements to troubleshoot specific areas in the code.
- Test the camera feed using OpenCV to ensure proper configuration.

## Contributors
- Vishnu Sreekumaran Nair
- Anusree Ambika Viswanathan
- Saumya Maurya
- Tanveer Singh

## Screenshots
Add relevant screenshots showing detection, alert display, and crowd monitoring features.

![Detection Screenshot](./snapshots/image1.png)  
*Alt text: Snapshot of people being detected with temperature readings displayed in bounding boxes.*

![Crowd Monitoring Screenshot](./snapshots/image2.png)  
*Alt text: Snapshot of Health Monitoring System showing crowd and temperature alerts with messages.*

## Acknowledgments
This project was developed as part of AI Project under the supervision of Professor Sanjeev Kumar for the Summer 2024 session.

## Contact
For questions or support, feel free to reach out to any team member listed in the Contributors section.
