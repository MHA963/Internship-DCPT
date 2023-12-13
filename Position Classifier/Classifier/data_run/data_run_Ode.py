from ultralytics import YOLO
import os
from pathlib import Path 
import cv2 as cv

trial_folder = Path('E:/Jasper/DBCGRT/Odense/BCCT HYPO Odense 03.2018') # folder to run through
new_save_folder = Path('E:/Jasper/DBCGRT/MHA/processed_images/Odense_processed') # folder to save images to 
model = YOLO('E:/Jasper/DBCGRT/MHA/Classifier/data_run/pos_classifier.pt')

armsUp_counter = 0
armsDown_counter = 0
sideView_counter = 0 
false_armsUp_counter = 0
false_armsDown_counter = 0
false_sideView_counter = 0

def prediction(image_path):
    return model.predict(source=image_path, imgsz=320)

def update_counters(predicted_label, image_name):
    global armsDown_counter, armsUp_counter, sideView_counter, false_sideView_counter, false_armsDown_counter, false_armsUp_counter

    if predicted_label == 'armsUp' and image_name.endswith('op'):
        armsUp_counter += 1
        #print(f"positive armsUp_counter incremented {armsUp_counter}")
    elif predicted_label == 'armsDown' and image_name.endswith('ned'):
        armsDown_counter += 1
        #print(f"positive armsDown_counter incremented {armsDown_counter}")
    elif predicted_label == 'Side' and image_name.endswith('side'):  # Update this number if necessary
        sideView_counter += 1
        #print(f"positive sideView_counter incremented {sideView_counter}")
    elif predicted_label == 'armsUp' and not image_name.endswith('op'):
        false_armsUp_counter += 1
        #print(f"negative armsUp_counter incremented {false_armsUp_counter}")
    elif predicted_label == 'armsDown' and not image_name.endswith('ned'):
        false_armsDown_counter += 1
        #print(f"negative arms_down_counter incremented {false_armsDown_counter}")
    elif predicted_label == 'Side' and not image_name.endswith('side'):  # Update this number if necessary
        false_sideView_counter += 1
        #print(f"Negative side_view_counter incremented {false_sideView_counter}")

def process_image(image_path, new_save_folder):
    image_name = Path(image_path).stem.lower()
    results = prediction(image_path)
    for r in results: 
        prediction_index = r.probs.top1
    
    class_mapping ={0: 'armsDown', 1: 'armsUp', 2:'Side'}
    predicted_label = class_mapping[prediction_index]
    update_counters(predicted_label, image_name)
    #print(f"The prediction for this image is : {predicted_label}")
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

# Print counters and accuracy
TP = armsUp_counter + armsDown_counter + sideView_counter
TN = false_armsUp_counter + false_armsDown_counter + false_sideView_counter
total = TP + TN

accuracy = (TP / total) * 100
# Calculate accuracy using precision and recall
precision = TP / (armsUp_counter + false_armsUp_counter)  # Precision for 'armsUp'
recall = TP / (armsUp_counter + false_armsDown_counter)   # Recall for 'armsUp'

# Calculate F1 score
f1_score = 2 * (precision * recall) / (precision + recall)

print('True Counts:')
print('Arms Up:', armsUp_counter)
print('Arms Down:', armsDown_counter)
print('Side:', sideView_counter)
print('\nFalse Counts:')
print('False Arms Up:', false_armsUp_counter)
print('False Arms Down:', false_armsDown_counter)
print('False Side:', false_sideView_counter)
print(f'\nAccuracy: {accuracy:.2f}%')
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1_score)
print('Done!')
