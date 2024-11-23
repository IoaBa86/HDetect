import cv2
from ultralytics import solutions
from ultralytics import YOLO

import supervision as sv

detections = sv.Detections.from_ultralytics(2)

model = YOLO("human.pt")


model.predict(source = "1.jpg", show=True, save=True, conf=0.6, line_width=1)


detections = detections[detections.class_id == classes.index("2")]
print(len(detections))

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    model="yolo11m.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
    #classes=[2],  # If you want to count specific classes i.e person and car with COCO pretrained model.
    #show_in=True,  # Display in counts
    #show_out=True,  # Display out counts
     line_width=2,  # Adjust the line width for bounding boxes and text display
)
