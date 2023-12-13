from ultralytics import YOLO
from pathlib import Path
import cv2 as cv
import os

def prediction(image_path):
    """
    Perform prediction using the YOLO model on the given image.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    List: YOLO model prediction results.
    """
    return model.predict(source=image_path, imgsz=320)

def update_counters(predicted_label, image_name):
    """
    Update counters based on the predicted label and image name.

    Parameters:
    - predicted_label (str): Predicted label ('armsUp', 'armsDown', 'Side').
    - image_name (str): Name of the image.

    Returns:
    None
    """
    global armsDown_counter, armsUp_counter, sideView_counter, false_sideView_counter, false_armsDown_counter, false_armsUp_counter

    if predicted_label == 'armsUp' and image_name.endswith('a'):
        armsUp_counter += 1
    elif predicted_label == 'armsDown' and image_name.endswith('b'):
        armsDown_counter += 1
    elif predicted_label == 'Side' and image_name.endswith('s'):
        sideView_counter += 1
    elif predicted_label == 'armsUp' and not image_name.endswith('a'):
        false_armsUp_counter += 1
    elif predicted_label == 'armsDown' and not image_name.endswith('b'):
        false_armsDown_counter += 1
    elif predicted_label == 'Side' and not image_name.endswith('s'):
        false_sideView_counter += 1

def process_image(image_path, new_save_folder):
    """
    Process an image, predict its label, and save it to the corresponding folder.

    Parameters:
    - image_path (str): Path to the image file.
    - new_save_folder (Path): Path to the folder for saving processed images.

    Returns:
    None
    """
    image_name = Path(image_path).stem.lower()
    results = prediction(image_path)
    
    for r in results:
        prediction_index = r.probs.top1
    
    class_mapping = {0: 'armsDown', 1: 'armsUp', 2: 'Side'}
    predicted_label = class_mapping[prediction_index]
    update_counters(predicted_label, image_name)
    
    destination_folder = new_save_folder / predicted_label
    destination_folder.mkdir(parents=True, exist_ok=True)
    
    image = cv.imread(image_path)
    cv.imwrite(Path(destination_folder, Path(image_path).name).as_posix(), image)

def process_image_in_folders(folder_path, new_save_folder):
    """
    Process images in a folder and its subfolders.

    Parameters:
    - folder_path (Path): Path to the top-level folder.
    - new_save_folder (Path): Path to the folder for saving processed images.

    Returns:
    None
    """
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            if Path(name).suffix.lower() in ['.png', '.jpg']:
                image_path = Path(path, name).as_posix()
                process_image(image_path, new_save_folder)




if __name__ == '__main__':
    
    trial_folder = Path('E:/Jasper/DBCGRT/Aalborg/BCCT HYPO Aalborg 03.2018')  # Folder to run through
    new_save_folder = Path('E:/Jasper/DBCGRT/MHA/processed_images/Aalborg_processed')  # Folder to save images to
    model = YOLO('E:/Jasper/DBCGRT/MHA/Classifier/data_run/pos_classifier.pt')
    armsUp_counter = 0
    armsDown_counter = 0
    sideView_counter = 0
    false_armsUp_counter = 0
    false_armsDown_counter = 0
    false_sideView_counter = 0
    
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
