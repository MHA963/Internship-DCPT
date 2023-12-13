import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def plot_sternal_points(dataframe: pd.DataFrame, index):
    img = plt.imread(dataframe['image_path'].loc[index])
    
    # Plot the bottom point
    plt.scatter(int(np.round(dataframe['scale_x'].loc[index])),
                int(np.round(dataframe['scale_y'].loc[index])),
                c='purple', s=25, marker='o')
    
    # Plot the top point
    plt.scatter(int(np.round(dataframe['sternal_notch_x'].loc[index])),
                int(np.round(dataframe['sternal_notch_y'].loc[index])),
                c='green', s=25, marker='o')

    distance = dataframe['distance'].loc[index]
    plt.text(10, 10, f'Distance: {distance}', color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))

    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    file_path = Path("C:/Users/student/Desktop/images/NMS_based_df.csv")
    dataframe = pd.read_csv(file_path, index_col=0)

    # Specify the desired distance
    desired_distance = 16.19


    # Find indices where the distance matches the desired value
    matching_indices = dataframe[dataframe['distance'] == desired_distance].index

    # Plot images for each matching index
    for index in matching_indices:
        plot_sternal_points(dataframe, index)

    print('Done')
