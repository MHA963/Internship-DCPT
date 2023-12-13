import os
import re
"""Randomization number extracted from image name"""
# Path to the folder containing patient images
image_folder = 'C:/Users/student/Desktop/images/Test/Side'

# List all files in the image folder
image_files = os.listdir(image_folder)

# Define a case-insensitive regex pattern to extract randomization numbers and dates from image names
pattern = re.compile(r'HYPO-(\d+)_(\d+)_s', re.IGNORECASE)

# Create a list to store randomization numbers and dates
randomization_info = []

for image_file in image_files:
    # Extract randomization number and date from the image name using regex
    match = pattern.search(image_file)
    if match:
        randomization_number = match.group(1)
        date = match.group(2)
        randomization_info.append((randomization_number, date))

# Save the information to a text file
output_file = 'output.txt'

with open(output_file, 'w') as f:
    for randomization_number, date in randomization_info:
        f.write(f"{randomization_number} - {date}\n")

print("Randomization information exported to", output_file)
