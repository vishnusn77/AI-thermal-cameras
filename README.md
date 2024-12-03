# Real-Time Thermal and Crowd Monitoring System

This project is designed to monitor public health and manage crowd density in real-time using advanced thermal cameras, computer vision techniques, and dynamic alert displays. The system extracts temperature data from video feeds using OCR, tracks individual temperatures, and displays dynamic health and safety messages on external monitors.

---

## Features

- **Real-time Temperature Detection**: Extract temperature readings from video frames captured by thermal cameras.
- **Crowd Density Estimation**: Track the number of people in the monitored area and adjust occupancy limits accordingly.
- **Dynamic Health and Safety Alerts**: Display real-time alerts and warnings about temperature and crowd density on external monitors.
- **Camera Integration**: Works with HK Provix thermal cameras and uses Tesseract OCR for temperature data extraction.

---

## Requirements

### Hardware
- **HK Provix Camera**: A thermal camera with IVMS 4200 software integration.
  - **Camera 01**: Provides normal video with temperature readings.
  - **Camera 02**: Provides thermal camera feed.
- **External Monitor/Signboard**: For displaying health and safety alerts.

### Software
- **Python 3.8 or higher**.
- Dependencies specified in the `requirements.txt` file.

---

## Setup Instructions

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/vishnusn77/AI-thermal-cameras.git
cd AI-thermal-cameras
```

### Install Dependencies

Install the required dependencies listed in `requirements.txt` using the following command:

```bash
pip install -r requirements.txt
```

### Configure the Thermal Camera

Ensure the following camera setup is done:

- **Camera 01**: Provides normal video with temperature readings.
- **Camera 02**: Provides thermal camera feed.
- **RTSP URL for Camera 01**: `rtsp://admin:Admin12345@192.168.1.142/Streaming/channels/2`.

---

## File Overview

The project consists of the following main Python files:

1. **`people_detection.py`**
   - Contains the code to detect people in video frames.
   - Uses computer vision techniques such as object detection and tracking to identify and track individuals in real time.

2. **`sort.py`**
   - Contains the SORT (Simple Online and Realtime Tracking) algorithm.
   - Assigns unique IDs to people detected in the video frames, associating temperatures with specific individuals across frames.

3. **`app.py`**
   - Main script to run the application.
   - Integrates the temperature extraction system, crowd density estimation, and alert system.
   - Captures frames from the video feed, extracts temperatures using OCR, tracks individuals, and displays alerts if necessary.

4. **`alert_display.py`**
   - Responsible for generating and displaying health and safety alerts on external monitors.
   - Uses the extracted temperature and occupancy data to display real-time warnings related to temperature and crowd density.

---

## Running the Application

To start the system, run the following command:

```bash
python app.py
```

### What Happens When You Run the Application:

#### Temperature Extraction:

- The application captures frames from the video feed at regular intervals (every 0.25 seconds).
- It processes these images to isolate temperature values displayed in green color using OpenCV.
- Tesseract OCR is used to extract the temperature readings from the captured images.

#### Crowd Density Estimation:

- The system tracks the number of people in the frame using the SORT algorithm in `sort.py`, which assigns unique IDs to each detected person.
- If the number of people exceeds the threshold, the system will trigger a crowd density warning.

#### Alert System:

Based on the extracted temperatures and crowd density, the system will display real-time alerts on an external monitor connected via HDMI. These alerts include:

- **High Temperature Warning**: If the temperature exceeds the defined threshold (e.g., 37.5°C).
- **Crowd Density Warning**: If the number of people exceeds the occupancy limit.

---

## File Structure

- **`people_detection.py`**: Contains code for detecting people in video frames using object detection and tracking.
- **`sort.py`**: Implements the SORT algorithm for tracking people across frames and assigning unique IDs.
- **`app.py`**: Main script that controls the application, integrates people detection, temperature extraction, and alert systems.
- **`alert_display.py`**: Manages displaying health and safety alerts on external monitors.
- **`requirements.txt`**: Lists all required Python libraries for the project.
- **`README.md`**: Documentation for setting up and using the project.
- **`LICENSE`**: License information for the project.

---

## Example Alerts

- **High Temperature Detected**: Alerts the user when a temperature exceeds the predefined threshold.
- **Crowd Density High**: Alerts when the number of people exceeds the allowed occupancy limit.
- **Health and Safety Advisory**: Displays a general advisory message based on the monitored data.

---

## Troubleshooting

- **Dependencies Not Installed**: Ensure all required libraries are installed by running `pip install -r requirements.txt`.
- **Camera Not Detected**: Double-check the RTSP URL configuration and network connection to the camera.
- **OCR Errors**: Ensure Tesseract OCR is installed and properly configured. If necessary, update the Tesseract path in `app.py`.

---

## Customization

### Temperature Thresholds
You can customize the temperature detection threshold and other parameters within the `app.py` file. For example:

```python
temperature_threshold = 37.5  # Example threshold for high temperature
```

### Crowd Density Limits
Adjust the crowd density limit to set occupancy restrictions:

```python
max_occupancy = 50  # Example max occupancy
```

---

## Contributing

We welcome contributions to improve this project. To contribute:

1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request for review.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact

For any questions or support, please reach out to the project team:

- **Vishnu Sreekumaran Nair**
- **Anusree Ambika Viswanathan**
- **Saumya Maurya**
- **Tanveer Singh**


---

## Acknowledgments

This project was developed as part of Group Assignment 2 for the Summer 2024 session under the supervision of Professor Sanjeev Kumar.

---

## Screenshots

(Add a few screenshots or images demonstrating the system in action.)