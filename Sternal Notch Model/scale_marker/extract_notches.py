import numpy as np
import cv2
import torch
from ultralytics import YOLO
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def predict_on_image(image_path, model, threshold=0.3):
    """
    Make predictions on the given image using the YOLO model.

    Parameters:
    - image_path (str): Path to the image file.
    - model (YOLO): YOLO model object.
    - threshold (float): Confidence threshold for predictions.

    Returns:
    Tuple[List[List[float]], List[float]]: List of bounding boxes and corresponding confidence scores.
    """
    # Make predictions for the current image
    predictions = model.predict(source=image_path, imgsz=640, save_conf=True, save=False, verbose=False)

    # Initialize lists to store bounding boxes and confidence scores
    boxes_list = []
    scores_list = []

    # Iterate through predictions for the current image
    for prediction in predictions:
        boxes = prediction.boxes.data.tolist()
        scores = prediction.boxes.conf.tolist()

        # Append filtered boxes and scores to the lists
        boxes_list.extend([box[:4] for box in boxes])
        scores_list.extend(scores)

    return boxes_list, scores_list

def calculate_angle(sternal_notch_x1, sternal_notch_y1, scale_x, scale_y):
    """
    Calculate the angle between two points.

    Parameters:
    - sternal_notch_x1, sternal_notch_y1, scale_x, scale_y (float): Coordinates of points.

    Returns:
    float: Calculated angle in degrees.
    """
    # Check if any value is None, and return a default angle (you can adjust this value)
    if None in (sternal_notch_x1, sternal_notch_y1, scale_x, scale_y):
        return None  # Adjust this default angle as needed

    # Calculate angle if all values are not None
    angle_rad = np.arctan2(sternal_notch_y1 - scale_y, sternal_notch_x1 - scale_x)
    return np.degrees(angle_rad)

def process_sternal_notches(boxes, scores, image_width, image_height):
    """
    Process sternal notches to find the best pair and calculate the angle.

    Parameters:
    - boxes (List[List[float]]): List of bounding boxes.
    - scores (List[float]): Confidence scores.
    - image_width (int): Width of the image in pixels.
    - image_height (int): Height of the image in pixels.

    Returns:
    Tuple[float, float, float, float, float, int, float, float]: Coordinates, angle, and additional information.
    """
    # Check the number of detections after NMS
    num_detections = len(boxes)
    
    #Uncomment the following code if u dont wanna use nms and calculate the sternal notch based on the angle
    # if num_detections > 2:
        
    #     if not boxes: 
    #         print("No sternal notch detected")
    #         sternal_notch_x1 = sternal_notch_y1 = sternal_notch_x2 = sternal_notch_y2 = angle = distance = None
    #         flag = 0  # Anything else
    #     else: 
    #         # Calculate angles for all sternal notches
    #         best_angle_diff = float('inf')
    #         best_sternal_notch_indices = None

    #         # Iterate through all pairs of sternal notches
    #         for i in range(num_detections):
    #             for j in range(i + 1, num_detections):
    #                 angle = calculate_angle(
    #                     (boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2,
    #                     (boxes[j][0] + boxes[j][2]) / 2, (boxes[j][1] + boxes[j][3]) / 2
    #                 )
                    
    #                 # Check if the angle is close to 90 degrees
    #                 angle_diff = abs(angle - 90)

    #                 if angle_diff < best_angle_diff:
    #                     best_angle_diff = angle_diff
    #                     best_sternal_notch_indices = (i, j)

    #         if best_sternal_notch_indices: 
    #             # Extract the best pair of sternal notches
    #             i, j = best_sternal_notch_indices
    #             # Sort the boxes based on y-coordinate
    #             sorted_boxes = sorted([boxes[i], boxes[j]], key=lambda box: -box[1])
    #             # Calculate the sternal notches
    #             sternal_notch_x1, sternal_notch_y1, sternal_notch_x2, sternal_notch_y2 = calculate_sternal_notch(sorted_boxes)
    #             # Calculate angle for the best pair
    #             angle = calculate_angle( sternal_notch_x1, sternal_notch_y1, sternal_notch_x2, sternal_notch_y2)

    #             flag = 3  # Corrected two detections based on the closest angle to 90 degrees
    #         else: 
    #             print("No valid sternal notch pair detected")
    #             sternal_notch_x1 = sternal_notch_y1 = sternal_notch_x2 = sternal_notch_y2 = angle = None
    #             flag = 0  # Anything else
                
    # Flag images based on the number of detections
    if num_detections == 2:
        # Sort the boxes based on y-coordinate
        sorted_boxes = sorted(boxes, key=lambda box: -box[1])
        sternal_notch_x1, sternal_notch_y1, sternal_notch_x2, sternal_notch_y2 = calculate_sternal_notch(sorted_boxes)

        angle = calculate_angle( sternal_notch_x1, sternal_notch_y1, sternal_notch_x2, sternal_notch_y2)
        confidence_A = scores[0]
        confidence_B = scores[1]
        distance = calculate_distance_percentage( sternal_notch_x1, sternal_notch_y1, sternal_notch_x2, sternal_notch_y2, image_width, image_height)

        # Calculate distance for the best pair:
        flag = 2  # Correct: two detections with the correct order

    elif num_detections == 1:
        # Assign the single sternal notch to sternal_notch_x1 and sternal_notch_y1
        x1, y1, x2, y2 = boxes[0]
        sternal_notch_x1 = (x1 + x2) / 2
        sternal_notch_y1 = (y1 + y2) / 2
        confidence_A = scores[0]
        

        # Placeholder values for the second sternal notch
        sternal_notch_x2 = sternal_notch_y2 = distance = angle = None
        confidence_B = None
        flag = 1  # One detection

    else:
        # Placeholder values for cases with 0 detections
        sternal_notch_x1 = sternal_notch_y1 = sternal_notch_x2 = sternal_notch_y2 = angle = distance = confidence_A = confidence_B = None
        flag = 0  # Anything else

    return sternal_notch_x1, sternal_notch_y1, sternal_notch_x2, sternal_notch_y2, angle, flag, distance, confidence_A,confidence_B

def calculate_distance_percentage(x1, y1, x2, y2, image_width, image_height):
    """
    Calculate the distance between two points as a percentage of the image size.

    Parameters:
    - x1, y1, x2, y2 (float): Coordinates of two points.
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    float: Calculated distance as a percentage.
    """
    # Convert coordinates to percentages
    x1_percent = (x1 / image_width) * 100.0
    y1_percent = (y1 / image_height) * 100.0
    x2_percent = (x2 / image_width) * 100.0
    y2_percent = (y2 / image_height) * 100.0

    # Calculate the Euclidean distance
    euclidean_distance = np.sqrt((x2_percent - x1_percent)**2 + (y2_percent - y1_percent)**2)

    return euclidean_distance

def calculate_sternal_notch(box_list):
    """
    Calculate sternal notch coordinates based on a list of bounding boxes.

    Parameters:
    - box_list (List[List[float]]): List of bounding boxes.

    Returns:
    Tuple[float, float, float, float]: Calculated sternal notch coordinates.
    """
    if len(box_list) == 2 :  
        x1, y1, x2, y2 = box_list[0]
        sternal_notch_x1 = (x1 + x2) / 2
        sternal_notch_y1 = (y1 + y2) / 2

        # Assign the second sternal notch to sternal_notch_x2 and sternal_notch_y2
        x1, y1, x2, y2 = box_list[1]
        sternal_notch_x2 = (x1 + x2) / 2
        sternal_notch_y2 = (y1 + y2) / 2     
        return sternal_notch_x1, sternal_notch_y1,sternal_notch_x2, sternal_notch_y2

    elif len(box_list) ==1 :
        x1, y1, x2, y2 = box_list[0]
        sternal_notch_x1 = (x1 + x2) / 2
        sternal_notch_y1 = (y1 + y2) / 2
        return sternal_notch_x1, sternal_notch_y1, None, None
def apply_nms(boxes, scores, confidence_threshold=0.3, nms_threshold=0.3):
    """
    Apply Non-Maximum Suppression (NMS) to eliminate redundant bounding boxes.

    Parameters:
    - boxes (List[List[float]]): List of bounding boxes.
    - scores (List[float]): Confidence scores.
    - confidence_threshold (float): Confidence threshold.
    - nms_threshold (float): Intersection over Union (IoU) threshold.

    Returns:
    Tuple[List[List[float]], List[float]]: Filtered bounding boxes and corresponding confidence scores.
    """
    # Convert boxes and scores to NumPy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Get indices of boxes to keep after NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, nms_threshold)

    # Extract filtered boxes and scores using indices
    filtered_boxes = [boxes[i] for i in indices]
    filtered_scores = [scores[i] for i in indices]

    return filtered_boxes, filtered_scores

if __name__ == "__main__":
    """
    This script processes a set of image folders using a pre-trained YOLO model to detect sternal notches.
    It calculates angles between sternal notches and filters the results based on specified criteria.
    The processed data is then stored in a Pandas DataFrame and saved to a CSV file.

    Steps:
    1. Load the YOLO model from the specified path.
    2. Define the top-level folders containing images to be processed.
    3. Initialize an empty Pandas DataFrame to store the results.
    4. Iterate through each specified top-level folder and its subfolders.
    5. For each image file in the folders, use the predict_on_image function to get bounding boxes and scores.
    6. Process the sternal notches using the process_sternal_notches function.
    7. Append the results to the Pandas DataFrame.
    8. Count occurrences of each flag category indicating the number of sternal notches detected.
    9. Display the flag counts.
    10. Save the processed data in the DataFrame to a CSV file named 'NMS_based_df_procent.csv'.

    Note: Make sure to replace the YOLO model path and top-level folder paths with the correct paths for your setup.
    """
    # Load the YOLO model
    model = YOLO('C:/Users/student/Desktop/images/scale_model/runs/detect/train/weights/best.pt')
    # Define the top-level folder containing images
    top_folders = [
                ("Aalborg", "E:/Jasper/DBCGRT/MHA/Module_test/Aalborg/armsUp"),
                ("Aarhus", "E:/Jasper/DBCGRT/MHA/Module_test/Aarhus/armsUp"),
                ("Odense", "E:/Jasper/DBCGRT/MHA/Module_test/Odense/armsUp"),
                ("Vejle", "E:/Jasper/DBCGRT/MHA/Module_test/Vejle/armsUp"),
                ("Dresden", "E:/Jasper/DBCGRT/MHA/Module_test/Dresden/armsUp"),            
    ]
    
    # Initialize the dataframe
    df_results = pd.DataFrame(columns=['center', 'image_name', 'image_path',
                                    'sternal_notch_x', 'sternal_notch_y',
                                    'scale_x', 'scale_y', 'Angle', 'flag', 'distance','sternal_notch_conf', 'scale_conf'])

    for center_name, top_folder_path in top_folders:
    # Iterate through images in the top folders
        for root, dirs, files in os.walk(top_folder_path):
            for file in tqdm(files, desc=f"processing {center_name}", unit="image"):
            # Check if the file is an image
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)

                    # Use your predict_on_image function to get a list of boxes and scores
                    boxes, scores = predict_on_image(image_path, model)
                    boxes, scores = apply_nms(boxes, scores)
                    # Get image dimensions
                    image = cv2.imread(image_path)
                    image_height, image_width, _ = image.shape 
                     
                    sternal_notch_x1, sternal_notch_y1, sternal_notch_x2, sternal_notch_y2, angle, flag, distance, sternal_notch_conf, scale_conf = process_sternal_notches(boxes, scores, image_height, image_width)
                    
                    # Append the results to the dataframe including the center name
                    new_row = {
                        'center': center_name,
                        'image_name': file,
                        'image_path': image_path,
                        'sternal_notch_x': sternal_notch_x1,
                        'sternal_notch_y': sternal_notch_y1,
                        'scale_x': sternal_notch_x2,
                        'scale_y': sternal_notch_y2,
                        'Angle': angle,
                        'flag': flag,
                        'distance': distance,
                        'sternal_notch_conf': sternal_notch_conf,
                        'scale_conf': scale_conf 
                    }
                    df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)


    # Count occurrences of each flag category
    flag_counts = df_results['flag'].value_counts()

    # Display the flag counts
    print("Flag Counts:")
    print("Flag 0 count:", flag_counts.get(0, 0))
    print("Flag 1 count:", flag_counts.get(1, 0))
    print("Flag 2 count:", flag_counts.get(2, 0))
    print("Flag 3 count:", flag_counts.get(3, 0))

    # Save the dataframe to a CSV file
    df_results.to_csv('NMS_based_df_procent.csv', index=True)
    print("Done")
