import numpy as np 
import cv2
import torch 
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os 
from pathlib import Path

model = YOLO('E:/Jasper/DBCGRT/MHA/Classifier/runs/classify/train/&weights/best.pt')

top_folder= "E:/Jasper/DBCGRT/Dataset/Aalborg/BCCT HYPO Aalborg 03.2018"

def find_image_files(folder):
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith((".JPG", ".png", ".jpeg")):
                image_files.append(os.path.join(root,file))
    
    return image_files


def plot_boxes (results):
    xyxys =[]
    confidences =[]
    class_ids = []
    
    for result in results :
        boxes =result.boxes.cpu().numpy()
        xyxys.append(boxes.xyxy)
        confidences.append(boxes.conf.item())
        class_ids.Append(boxes.cls.item())
        
    return results[0].plot,xyxys,confidences,class_ids

 
image_files = find_image_files(top_folder)

for img_path in image_files: 
    results = model.predict(source=img_path, imgsz=320, save_conf=True, save =True)
    for r in results: 
        print(f"{r.probs}")
    #plot_boxes(results)    
    
    


        