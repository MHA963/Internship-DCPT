from ultralytics import YOLO
import os
from pathlib import Path 
import cv2 as cv

trial_folder = Path('E:/Jasper/DBCGRT/Aarhus/BCCT HYPO Aarhus 03.2018')  # Folder to run through
model = YOLO('E:/Jasper/DBCGRT/MHA/Classifier/data_run/pos_classifier.pt')

armsUp_counter = 0
armsDown_counter = 0
sideView_counter = 0 
false_armsUp_counter = 0
false_armsDown_counter = 0
false_sideView_counter = 0

def prediction(image_path):
    """
    Make predictions using the YOLO model.

    Parameters:
        image_path (str): The path to the image for prediction.

    Returns:
        List: A list of prediction results.
    """
    return model.predict(source=image_path, imgsz=320)

def update_counters(predicted_label, image_name):
    """
    Update counters based on predicted label and image name.

    Parameters:
        predicted_label (str): The predicted label from the model.
        image_name (str): The name of the image.

    Returns:
        None
    """
    global armsDown_counter, armsUp_counter, sideView_counter, false_sideView_counter, false_armsDown_counter, false_armsUp_counter

    if predicted_label == 'armsUp' and image_name.endswith('(1)'):
        armsUp_counter += 1
    elif predicted_label == 'armsDown' and image_name.endswith(''):
        armsDown_counter += 1
    elif predicted_label == 'Side' and image_name.endswith('(2)'):  # Update this number if necessary
        sideView_counter += 1
    elif predicted_label == 'armsUp' and not image_name.endswith('(1)'):
        false_armsUp_counter += 1
    elif predicted_label == 'armsDown' and not image_name.endswith(''):
        false_armsDown_counter += 1
    elif predicted_label == 'Side' and not image_name.endswith('(2)'):  # Update this number if necessary
        false_sideView_counter += 1

def process_image(image_path):
    """
    Process an image, make predictions, and update counters.

    Parameters:
        image_path (str): The path to the image.

    Returns:
        None
    """
    image_name = Path(image_path).stem.lower()
    results = prediction(image_path)
    for r in results: 
        prediction_index = r.probs.top1
    
    class_mapping ={0: 'armsDown', 1: 'armsUp', 2:'Side'}
    predicted_label = class_mapping[prediction_index]
    update_counters(predicted_label, image_name)

def process_image_in_folders(folder_path):
    """
    Process all images in a folder and its subfolders.

    Parameters:
        folder_path (Path): The path to the folder containing images.

    Returns:
        None
    """
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            if Path(name).suffix.lower() in ['.png', '.jpg']:
                image_path = Path(path, name).as_posix()
                process_image(image_path)

process_image_in_folders(trial_folder)       

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

# Save results to a text file
with open('Aarhus_results.txt', 'w') as file:
    file.write('True Counts:\n')
    file.write(f'Arms Up: {armsUp_counter}\n')
    file.write(f'Arms Down: {armsDown_counter}\n')
    file.write(f'Side: {sideView_counter}\n\n')
    file.write('False Counts:\n')
    file.write(f'False Arms Up: {false_armsUp_counter}\n')
    file.write(f'False Arms Down: {false_armsDown_counter}\n')
    file.write(f'False Side: {false_sideView_counter}\n\n')
    file.write(f'Accuracy: {accuracy:.2f}%\n')
    file.write(f'Precision: {precision}\n')
    file.write(f'Recall: {recall}\n')
    file.write(f'F1 Score: {f1_score}\n')

print('Results saved to prediction_results.txt')
