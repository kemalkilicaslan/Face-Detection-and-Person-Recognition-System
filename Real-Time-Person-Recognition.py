# Real Time Person Recognition

import cv2  # importing the OpenCV library
from ultralytics import YOLO  # importing the YOLO algorithms

# Install custom YOLO model
model = YOLO('Kemal-Kilicaslan.pt')

# Turn on the camera
cap = cv2.VideoCapture(0)

# Create a loop for continuous image acquisition and processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make predictions using the model
    results = model.predict(source=frame, show=False)

    # Draw the prediction results in the frame
    annotated_frame = results[0].plot()

    # Show annotated frame and set window title
    cv2.imshow('Kemal-Kilicaslan', annotated_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close pop-up windows
cap.release()
cv2.destroyAllWindows()