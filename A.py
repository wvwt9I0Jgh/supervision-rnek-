import cv2
import supervision as sv
from ultralytics import YOLO    

model = YOLO('yolov8n.pt')

img = cv2.imread('bus.jpg')
results = model(img)[0]
detections = sv.Detections.from_ultralytics(results)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

img_boxes = box_annotator.annotate(scene=img, detections=detections)
img_labels = label_annotator.annotate(scene=img_boxes, detections=detections)

cv2.imshow('image', img_labels)
cv2.waitKey(0)
cv2.destroyAllWindows()