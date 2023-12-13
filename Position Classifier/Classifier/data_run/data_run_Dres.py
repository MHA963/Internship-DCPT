from ultralytics import YOLO
import os
from pathlib import Path 
import cv2 as cv

trial_folder = Path('E:/Jasper/DBCGRT/Dresden CGC/BCCT HYPO Dresden CGC 03.2018')  # folder to run through
new_save_folder = Path('E:/Jasper/DBCGRT/MHA/processed_images/Dresden_processed')  # folder to save images to 
model = YOLO('E:/Jasper/DBCGRT/MHA/Classifier/data_run/pos_classifier.pt')

def prediction(image_path):
    return model.predict(source=image_path, imgsz=320)

def process_image(image_path, new_save_folder):
    image_name = Path(image_path).stem.lower()
    results = prediction(image_path)
    for r in results: 
        prediction_index = r.probs.top1
    
    class_mapping = {0: 'armsDown', 1: 'armsUp', 2: 'Side'}
    predicted_label = class_mapping[prediction_index]
    print(f"The prediction for this image is: {predicted_label}")
    destination_folder = new_save_folder / predicted_label
    destination_folder.mkdir(parents=True, exist_ok=True)
    image = cv.imread(image_path)
    cv.imwrite(Path(destination_folder, Path(image_path).name).as_posix(), image)

def process_image_in_folders(folder_path, new_save_folder):
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            if Path(name).suffix.lower() in ['.png', '.jpg']:
                image_path = Path(path, name).as_posix()
                process_image(image_path, new_save_folder)

process_image_in_folders(trial_folder, new_save_folder)       
print('Done!')
