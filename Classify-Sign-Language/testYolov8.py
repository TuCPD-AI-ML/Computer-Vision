from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO('lastv8.pt')
result = model.predict(source='0', show = True)