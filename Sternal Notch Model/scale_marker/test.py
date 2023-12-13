import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def plot_sternal_points(dataframe: pd.DataFrame, index):
    img_path = dataframe['image_path'].loc[index]
    scale_x, scale_y = dataframe['scale_x'].loc[index], dataframe['scale_y'].loc[index]
    sternal_notch_x, sternal_notch_y = dataframe['sternal_notch_x'].loc[index], dataframe['sternal_notch_y'].loc[index]

    if not (np.isnan(scale_x) or np.isnan(scale_y)):
        img = plt.imread(img_path)
        plt.scatter(int(np.round(scale_x)), int(np.round(scale_y)), c='purple', s=25, marker='o')

    if not (np.isnan(sternal_notch_x) or np.isnan(sternal_notch_y)):
        plt.scatter(int(np.round(sternal_notch_x)), int(np.round(sternal_notch_y)), c='green', s=25, marker='o')
    
    distance = dataframe['distance'].loc[index]
    plt.text(10, 10, f'Distance: {distance}', color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))
    
    if 'img' in locals():
        plt.imshow(img)
        plt.show()

def prompt_user():
    while True:
        response = input("Do you want to try again? (y/n): ").lower()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print("Invalid response. Please enter 'y' or 'n'.")

if __name__ == "__main__":
    dataframe = pd.read_csv(Path("C:/Users/student/Desktop/images/NMS_based_df.csv"), index_col=0)

    while True:
        random_int = np.random.randint(0, len(dataframe))
        try:
            # Check the number of sternal notches
            num_sternal_notches = dataframe[['sternal_notch_x', 'sternal_notch_y']].loc[random_int].count()
            plot_sternal_points(dataframe, random_int)
            
            if num_sternal_notches == 2:
                print("Two sternal notches found.")
            elif num_sternal_notches == 1:
                print("Only one sternal notch found.")
            else:
                print("No sternal notches found.")
                if not prompt_user():
                    break  # Exit the loop if the user chooses not to try again
            
            if not prompt_user():
                break  # Exit the loop if the user chooses not to try again
        except ValueError as e:
            print(f"Skipping index {random_int} due to error: {e}")
            if not prompt_user():
                break  # Exit the loop if the user chooses not to try again

    print('Done')
