from people_detection import detect_and_track_people

# Main function
def run(video_source="webcam", video_path=None, rtsp_url=None):
    detect_and_track_people(video_source=video_source, video_path=video_path, rtsp_url=rtsp_url)

if __name__ == "__main__":

    # For live webcam footage
    # run(video_source="webcam")

    # For a video file (replace with the actual path to your video file)
    # run(video_source="video", video_path="./sample/footage.mp4")

    # For RTSP stream from the Provix camera
    run(video_source="rtsp", rtsp_url="rtsp://admin:Admin12345@192.168.1.142/Streaming/channels/2")
