from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor

#model = YOLO("Salinanbest.pt")
model = YOLO("best_4.pt")
model.predict(source="cctv1.mp4" , show=True,show_labels=False, line_width=2)