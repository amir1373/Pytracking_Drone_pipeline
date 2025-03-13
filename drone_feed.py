import cv2
import socket
from pytracking.tracker.dimp import DiMP
from pytracking.evaluation import Tracker

# Global variables for UDP communication
udp_ip = "127.0.0.1"  # Change to your desired IP
udp_port = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def initialize_tracker():
    """Initialize the DiMP tracker with proper parameter loading."""
    tracker = Tracker('dimp', 'super_dimp')
    return tracker.tracker_class(tracker.get_parameters())

def main():
    # Initialize the tracker
    tracker_instance = initialize_tracker()

    # Open a connection to the video stream URL (replace with your URL)
    video_url = "http://127.0.0.1:5000/video_feed"  # Example URL
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(f"Error: Could not open video stream from URL: {video_url}")
        return

    # Read the first frame from the video stream
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video stream.")
        cap.release()
        return

    # Create a named window for ROI selection so that it's explicitly visible and resizable
    cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
    # Let the user select the initial bounding box (ROI)
    init_state = cv2.selectROI('Select Object', frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow('Select Object')

    # Check if a valid ROI was selected
    if init_state == (0, 0, 0, 0):
        print("No ROI selected. Exiting.")
        cap.release()
        return

    # Initialize the tracker with the first frame and the selected bounding box
    tracker_instance.initialize(frame, {'init_bbox': init_state})

    while True:
        # Read a new frame from the video stream
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Track the object in the current frame
        state = tracker_instance.track(frame)

        # Retrieve and draw the bounding box
        bbox = state['target_bbox']
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

        # Send the bounding box results over UDP (center_x, center_y, width, height)
        center_x = int(bbox[0] + bbox[2] / 2)
        center_y = int(bbox[1] + bbox[3] / 2)
        width = int(bbox[2])
        height = int(bbox[3])

        message = f"{center_x},{center_y},{width},{height}"
        sock.sendto(message.encode(), (udp_ip, udp_port))

        # Display the frame with the tracking overlay
        cv2.imshow('Tracking', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
