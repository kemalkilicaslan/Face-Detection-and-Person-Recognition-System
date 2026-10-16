# Real Time Face Detection

import cv2  # importing the OpenCV library

# Install haarcascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Let's create a video stream for the camera.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # read a frame from the camera  

    # Let's convert it to gray scale image.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces on a greyscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the detected faces and save
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 255), 14)

    # Show video with faces detected
    cv2.imshow('frame',frame)

    # Switch off video when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all open windows
cap.release()
cv2.destroyAllWindows()