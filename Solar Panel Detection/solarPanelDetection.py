from typing import Any
from ultralytics import YOLO
import torch
import os
import numpy as np

import cv2 as cv

class SolarPanelDetection:
    def __init__(self, capture_index):
        print("__init__")
        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device} device')

        self.model = self.load_model()
    
    def load_model(self):
        #path for the model weights
        model_path = os.path.expanduser("/home/tpickup/projects/solar-panel-detection/Solar Panel Detection/runs/detect/train8/weights/best.pt")

        model = YOLO(model_path)
        #model.fuse()
        return model   
    
    
    def predict(self, frame):
        print("detecting...") 
        results = self.model(frame)

        #print(results)
        return results
    
    def plot_bboxes(self, frame, results):
        print("plotting boxes...")

        xyxys = []
        confidence = []
        class_ids = []

        #get pounding box prediction form frakem
        for result in results:
            boxes = result.boxes.cpu().numpy()


            #xyxys = boxes.xyxy
            #for xyxy in xyxys:
            #   cv.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            if len(boxes.xyxy) > 0:
                xyxys.append(boxes.xyxy)
                confidence.append(boxes.conf)
                class_ids.append(boxes.cls)
        
        return results[0].plot(), xyxys, confidence, class_ids, frame



        

    def get_panel_coordinates(self, frame): 

        pass

    def __call__(self):
        print("I have started")
        #load video
        in_frame_rate = 29.97
        resolution = (3840, 2160)
        frame_rate = in_frame_rate / 3
        cap = cv.VideoCapture("/home/tpickup/projects/solar-panel-detection/Solar Panel Detection/georeferencing/DJI_0753.MP4")
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter('output.avi', fourcc, frame_rate, resolution)

        counter = 0

        #create an isntance of SolarPanelDetection
        while cap.isOpened():
            counter += 1
            if counter % 3 == 0:
                continue

            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            #cv.imshow('frame', gray)
            
            results = self.predict(frame)

            bb_frame, xyxys, confidence, class_ids, frame = self.plot_bboxes(frame, results)
            print("xyxys: ", np.asarray(xyxys).shape)
            for xyxy, conf, id in zip(xyxys[0], confidence[0], class_ids[0]):
                frame = cv.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 5)
                frame = cv.rectangle(frame, (int(xyxy[0])+5, int(xyxy[1])-10), (int(xyxy[0])+400, int(xyxy[1])-30), (0, 255, 0), 20)
                frame = cv.putText(frame, "solar-panel: " + str(conf), (int(xyxy[0]), int(xyxy[1])-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

            cv.imshow('frame', frame)

            out.write(frame)

            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        out.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    print("__main__")
    solarPanelDetection = SolarPanelDetection(0)
    solarPanelDetection()
