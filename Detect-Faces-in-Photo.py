# Detect Faces in Photo

import cv2  # importing the OpenCV library

# Install haarcascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image
image = cv2.imread('How-I-Met-Your-Mother-Faces-Photo.jpg')

# Converting the image to grayscale (required for face detection)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecting faces in the grayscale image using the haarcascade classifier
# 1.3 is the scaleFactor (specifies how much the image size is reduced at each image scale)
# 5 is the minNeighbors (defines how many neighbors each candidate rectangle should have to retain it)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Drawing rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 4)

# Displaying the image with detected faces
cv2.imshow('HIMYM-faces-detector', image)

# Saving the image with detected faces
output = 'HIMYM-Faces-Detected-Photo.jpg'
cv2.imwrite(output, image)

# Waiting for a key press to close the image window
cv2.waitKey()
cv2.destroyAllWindows()
