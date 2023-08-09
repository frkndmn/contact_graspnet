import cv2
import cv2.aruco as aruco

# Load the camera matrix and distortion coefficients
camera_matrix = # Your camera matrix
dist_coeffs = # Your distortion coefficients

# Initialize the ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

# Initialize the detector parameters
params = aruco.DetectorParameters_create()

# Initialize the video capture device (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=params)

    # Draw the detected markers
    aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the resulting frame
    cv2.imshow('ArUco detection', frame)

    # Wait for the 'q' key to be pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture device and close the windows
cap.release()
cv2.destroyAllWindows()