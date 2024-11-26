from multiprocessing import Process, Queue
import time
from people_detection import detect_and_track_people
from alert_display import run_alert_system

# Function to handle the detection and populate alerts
def detection_process(alert_queue, video_source="video", video_path=None, rtsp_url=None):
    def alert_callback(alert_message, color):
        """
        Sends alert messages to the alert system.
        :param alert_message: The alert message.
        :param color: The color code for the alert text.
        """
        alert_queue.put({"message": alert_message, "color": color})

    # Assuming `detect_and_track_people` processes video in real-time and invokes the callback when needed
    detect_and_track_people(
        video_source=video_source,
        video_path=video_path,
        rtsp_url=rtsp_url,
        alert_callback=alert_callback  # Passing the alert callback
    )

# Main function to run the detection and alert display simultaneously
def main():
    # Queue to communicate between processes
    alert_queue = Queue()

    # Processes for detection and alert display
    detection_proc = Process(
        target=detection_process,
        args=(alert_queue, ),
        kwargs={
            "video_source": "video",  # Change to "webcam" or "rtsp" as needed
            "video_path": "./sample/footage_10.mp4",  # Or specify the RTSP URL for live video
            # Uncomment the following line for RTSP
            # "rtsp_url": "rtsp://admin:Admin12345@192.168.1.142/Streaming/channels/2",
        }
    )

    alert_display_proc = Process(target=run_alert_system, args=(alert_queue,))

    try:
        # Start both processes
        detection_proc.start()
        alert_display_proc.start()

        # Wait for both processes to complete
        while True:
            time.sleep(1)  # Main process stays alive without blocking

    except KeyboardInterrupt:
        print("Shutting down processes...")
        detection_proc.terminate()
        alert_display_proc.terminate()
        detection_proc.join()
        alert_display_proc.join()

if __name__ == "__main__":
    main()
