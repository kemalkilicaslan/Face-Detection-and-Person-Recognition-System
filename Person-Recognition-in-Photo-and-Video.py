# Person Recognition in Photo and Video

from ultralytics import YOLO

# Custom trained YOLO model
model = YOLO('How-I-Met-Your-Mother-Person-Recognition-model.pt')

# Processing source images with the model
results = model('How-I-Met-Your-Mother-Person-Photo.jpg', save=True) # for photo
# or
results = model('How-I-Met-Your-Mother-Person-Video.mp4', save=True) # for video