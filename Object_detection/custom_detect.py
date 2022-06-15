import os
import sys
from pathlib import Path
from turtle import width
import cv2 
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, time_sync
import numpy as np
from coco_classes import classes

from models.common import DetectMultiBackend

class Detector:
    def __init__(self,weights,data):
        self.weights = weights
        self.data = data
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.org = (50, 50)
        self.fontScale = 1
        self.color = (0, 255, 0 )
        self.thickness = 1
        self.visualize = True
        self.device = select_device('0')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=data, fp16=False)
        self.class_of_interest = 'car'

    def convert_mid_to_corner(self, x,y,w,h):
        x1 = (x-(w/2))
        y1 = (y-(h/2))
        x2 = x1 + w
        y2 = y1 + h
        return [x1,y1,x2,y2]

    def convert_to_int(self, width, height,line_point):
        x1,y1,x2,y2 = line_point
        x1 = int(x1*width)
        x2 = int(x2*width)
        y1 = int(y1*height)
        y2 = int(y2*height)
        return x1, y1, x2, y2

    def predict_img(self,img):
        img0 = img.copy()
        img = cv2.resize(img, (640,480))
         
        height, width, _  = img.shape
        height0, width0, _ = img0.shape
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
        img = img.reshape(1,height,width,3)
        img = img.transpose((0,3,1,2))
        img = img/255.0
        img = torch.from_numpy(img).to(self.device).float()
        pred = self.model(img,augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, None, True, max_det=1000)
        class_count = 0

        # Line thickness of 2 px
        for i, det in enumerate(pred):

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                x,y,w,h = line[1], line[2], line[3], line[4]
                line_point = self.convert_mid_to_corner(x,y,w,h)
                x1,y1,x2,y2 = self.convert_to_int(width0, height0,line_point)

                if cls in [classes[self.class_of_interest]] :
                    class_count+=1 

                    if self.visualize:
                        cv2.rectangle(img0,(x1, y1), (x2, y2),self.color,self.thickness)

            cv2.putText(img0, f"{self.class_of_interest} count: {class_count}", self.org, self.font, 
                    self.fontScale, self.color, self.thickness, cv2.LINE_AA)

        cv2.imshow('test',img0)
        cv2.waitKey(24)

    def predict_video(self,video):
        cap = cv2.VideoCapture(video)
        
        while cap.isOpened():
            ret, img = cap.read()
            self.predict_img(img)
            if 0xFF == ord('q'):
                break

            


def main():
    test_img = 'cars.jpg'
    video_path = 'traffic-video.mp4'
    detector = Detector('yolov5l.pt','coco.yaml')
    # img = cv2.imread(test_img)
    # img0 = cv2.imread(test_img)
    # detector.predict_img(img)
    detector.predict_video(video_path)

if __name__ == "__main__":
    main()
