# from ultralytics import YOLO
# model = YOLO('lastv8.pt')
# results = model(source=0, show=True, conf=0.6, save=False)

import torch
import keras
import numpy as np
import cv2
from ultralytics import YOLO
import time

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = self.load_model()
    
    def load_model(self):
        model = YOLO('lastv8.pt')
        model.fuse()

        return model
    def predict(self, frame):
        results = self.model(frame, conf=0.5)
        return results
    
    def get_boxes(self, results):
        xyxys = []
        confs = []
        class_id = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
        xyxy = boxes.xyxy
        # xyxys.append(boxes.xyxy)
        # confs.append(boxes.conf)
        # class_id.append(boxes.cls)
        # return results[0].plot(), xyxy
        return xyxy

    def classify(self, image, model, alp):
        image = cv2.resize(image, (28, 28))
        image = image.astype("float") / 255.0
        image = np.reshape(image, (1, 28, 28, 1))
        pred = model.predict(image)
        idx = np.argmax(pred)
        return alp[idx]
    
    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        alp = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
        model_classify = keras.models.load_model('sign_lang4_1.h5')
        cv2.imshow("image", cv2.imread('signimage.png'))
        white_img = cv2.imread('white_img.png')
        white_img = cv2.resize(white_img, (500, 500))
        text_position = 0
        start_time = None
        current_text = ''
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = self.predict(frame)
            xyxy = self.get_boxes(results)
            # print(len(xyxy))
            # print(xyxy)
            if len(xyxy) == 1:
                x, y, x_max, y_max = list(map(int, xyxy[0][:4]))

                roi = frame[y-5:y_max-5, x-5:x_max-5]
                roi = cv2.flip(roi, 1)

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                cv2.imshow('roi', roi)

                alpha = self.classify(blur, model_classify, alp)

                cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 255, 0), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, alpha, (0, 130), font, 5, (0, 0, 255), 2)

                if alpha != '':
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time > 4:
                        cv2.putText(white_img, alpha, (text_position, 100), font, 2, (0, 0, 255), 2)
                        current_text += alpha 
                        text_position += 35
                        start_time = None
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    space_width = cv2.getTextSize(' ', font, 2, 2)[0][0]
                    text_position += space_width
                    current_text += ' '
            cv2.imshow('Object Detection', frame)
            cv2.imshow('Text', white_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obj_detection = ObjectDetection(capture_index=0)
    obj_detection()

