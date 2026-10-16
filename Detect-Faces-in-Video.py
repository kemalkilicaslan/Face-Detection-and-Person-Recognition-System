# Detect Faces in Video

import cv2  # importing the OpenCV library

# Install haarcascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open video
video_capture = cv2.VideoCapture('How-I-Met-Your-Mother-Faces-Video.mp4')  # write the name of the video file here
output_file = 'HIMYM-faces-detected.mp4'  # extension the name and extension of the video file to be recorded

# Video features
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create the VideoWriter object to save the video file
output_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    ret, frame = video_capture.read()  # read video frame

    if not ret:
        break

    # Convert frame to greyscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces on a greyscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the detected faces and save
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 14)

    # Write the drawn frame to the video file to be saved
    output_video.write(frame)

    # Show video with faces detected
    cv2.imshow('Video', frame)

    # Switch off video when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all open windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
