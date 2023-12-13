import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm 
import traceback 


def calculate_bounding_box(center_x, center_y, box_width, box_height, img_width, img_height):
    """
    Calculate the center coordinates and dimensions of a bounding box relative to the image size.

    Args:
        center_x (int): X-coordinate of the center of the bounding box.
        center_y (int): Y-coordinate of the center of the bounding box.
        box_width (int): Width of the bounding box.
        box_height (int): Height of the bounding box.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        tuple: A tuple containing the center coordinates and dimensions of the bounding box
               in terms of percentages relative to the image size (x_procent, y_procent, bbox_width_procent, bbox_height_procent).
    """
    x_procent = center_x / img_width 
    y_procent = center_y / img_height
    bbox_width_procent = box_width / img_width
    bbox_height_procent = box_height /img_height
    
    
    return x_procent, y_procent, bbox_width_procent, bbox_height_procent


def plot_bounding_boxes(image, sternal_purple_x, sternal_purple_y, sternal_green_x, sternal_green_y, bbox_width, bbox_height):
    """
    Optional function
    Plot sternal notches and bounding boxes on an image.

    Args:
        image (numpy.ndarray): The input image on which sternal notches and bounding boxes will be visualized.
        sternal_purple_x (int): X-coordinate of the purple sternal notch.
        sternal_purple_y (int): Y-coordinate of the purple sternal notch.
        sternal_green_x (int): X-coordinate of the green sternal notch.
        sternal_green_y (int): Y-coordinate of the green sternal notch.
        bbox_width (int): Width of the bounding box.
        bbox_height (int): Height of the bounding box.
    """
    # Create a copy of the image to avoid modifying the original
    img_with_boxes = image.copy()

    # Calculate the center coordinates and dimensions of the bounding boxes
    bbox_x1_purple, bbox_y1_purple, bbox_width_procent, bbox_height_procent = calculate_bounding_box(
        sternal_purple_x, sternal_purple_y, bbox_width, bbox_height, img_with_boxes.shape[1], img_with_boxes.shape[0])
    
    bbox_x1_green, bbox_y1_green, bbox_width_procent, bbox_height_procent = calculate_bounding_box(
        sternal_green_x, sternal_green_y, bbox_width, bbox_height, img_with_boxes.shape[1], img_with_boxes.shape[0])

    # Plot the image
    plt.imshow(img_with_boxes)

    # Plot sternal notches 
    plt.scatter([sternal_purple_x, sternal_green_x], [sternal_purple_y, sternal_green_y], c='red', marker='o', s=50)

    # Plot bounding boxes
    #calculation for the top-lef coordinates and dimensions of the bounding box could be included in a function to ease up the work but hey... 
    
    plt.gca().add_patch(plt.Rectangle((bbox_x1_purple * img_with_boxes.shape[1] - bbox_width_procent * img_with_boxes.shape[1] / 2,
                                       bbox_y1_purple * img_with_boxes.shape[0] - bbox_height_procent * img_with_boxes.shape[0] / 2),
                                      bbox_width_procent * img_with_boxes.shape[1], bbox_height_procent * img_with_boxes.shape[0],
                                      linewidth=2, edgecolor='green', facecolor='none'))
    
    plt.gca().add_patch(plt.Rectangle((bbox_x1_green * img_with_boxes.shape[1] - bbox_width_procent * img_with_boxes.shape[1] / 2,
                                       bbox_y1_green * img_with_boxes.shape[0] - bbox_height_procent * img_with_boxes.shape[0] / 2),
                                      bbox_width_procent * img_with_boxes.shape[1], bbox_height_procent * img_with_boxes.shape[0],
                                      linewidth=2, edgecolor='purple', facecolor='none'))

    # Display the plot
    plt.show()


def process_image(index, dataframe: pd.DataFrame, save_dir):
    """
    Process an image, retrieve sternal notches positions, calculate bounding boxes, and save the results.

    Args:
        index (int): Index of the image in the dataframe.
        dataframe (pd.DataFrame): The dataframe containing image information.
        save_dir (str): Directory where the results will be saved.

    Notes:
        - Reads the image file from the provided dataframe.
        - Retrieves sternal notches positions from the dataframe.
        - Calculates bounding box coordinates as percentages of image dimensions.
        - Saves the processed image and corresponding label in the specified save directory.
    """
    #read file 
    img = plt.imread(Path(dataframe['repeated path'].loc[index]))

    #retrieve the sternal notches positions
    
    sternal_purple_x = int(np.round(dataframe['Scale x'].loc[index]))
    sternal_purple_y = int(np.round(dataframe['Scale y'].loc[index]))
    sternal_green_x = int(np.round(dataframe['Sternal notch x'].loc[index]))
    sternal_green_y = int(np.round(dataframe['Sternal notch y'].loc[index]))
    img_height, img_width, _ = img.shape

    # bounding box width and height
    bbox_width = 600  
    bbox_height = 600

    #Calculate the bounding boxes coordinates in procent
    x1_procent, y1_procent, bbox_width_procent, bbox_height_procent = calculate_bounding_box(
        sternal_purple_x, sternal_purple_y, bbox_width, bbox_height, img_width, img_height)

    x2_procent, y2_procent, bbox_width_procent, bbox_height_procent = calculate_bounding_box(
        sternal_green_x, sternal_green_y, bbox_width, bbox_height, img_width, img_height)

    img_filename = Path(dataframe['repeated path'].loc[index]).stem

    image_save_dir = os.path.join(save_dir, 'images')
    label_save_dir = os.path.join(save_dir, 'labels')

    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)
    
    ## plotting the images with the bounding boxes
    #Can be used to plot the sternal notch + the bounding boxes to check if the coordinates are correct
    #plot_bounding_boxes(img, sternal_purple_x, sternal_purple_y, sternal_green_x, sternal_green_y, bbox_width, bbox_height)

    #save the image
    plt.imsave(os.path.join(image_save_dir, f'{img_filename}.jpg'), img)
    #save the label 
    with open(os.path.join(label_save_dir, f'{img_filename}.txt'), 'w') as f:
        f.write(f"0 {x1_procent:.4f} "
                f"{y1_procent:.4f} "
                f"{bbox_width_procent:.4f} "
                f"{bbox_height_procent:.4f}\n"
                f"0 {x2_procent:.4f} "
                f"{y2_procent:.4f} "
                f"{bbox_width_procent:.4f} "
                f"{bbox_height_procent:.4f}\n")


if __name__ == "__main__":
    
    counts = 0
    dataframe = pd.read_csv(Path("E:/Jasper/BCCTCore2.0/df.csv"), index_col=0)
    save_dir = "C:/Users/student/Desktop/images/scale_marker/dataset"

    train_dir = os.path.join(save_dir, 'train')
    val_dir = os.path.join(save_dir, 'val')
    
    #create the train/val folder if not existed
    os.makedirs(train_dir,exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # loop through the dataframe 
    for index in tqdm(range(len(dataframe))):
        #split the data 90% -10% 
        is_train = np.random.rand() < 0.9
        img_save_dir = train_dir if is_train else val_dir
        try: 
            process_image(index,dataframe,img_save_dir)
        except Exception as e: 
            print(f" Error processing image at index {index}: {e} ")
            counts +=1

    print(f"Done with {counts} errors.")


