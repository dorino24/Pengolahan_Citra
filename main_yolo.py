from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2

model = YOLO("bestfix.pt")

results = model.predict(source="cctv1.mp4",show=True ,conf=0.3,show_conf=True,line_width=1)
