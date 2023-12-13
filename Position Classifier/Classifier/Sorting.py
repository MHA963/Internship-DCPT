import os
import shutil
import pandas as pd
import re

""" In this file we tested the sorting algorithms for each hospital, if u need to work on a single studie,
    you can use this to start with, then edit the image_extraction module.
"""
    
    
def load_image_info(file_path):
    """
    Load randomization number and date information from a text file.

    Parameters:
    - file_path (str): Path to the text file.

    Returns:
    - dict: Dictionary mapping randomization numbers to lists of dates.
    """
    image_info = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Extracting information using regular expressions
            match = re.search(r' PatientBD: ([a-zA-Z]+ \d{6}) , Randomization Number: (\d+)', line)
            if match:
                patientBD, randomization = match.groups()
                if randomization not in image_info:
                    image_info[patientBD] = []
                image_info[patientBD].append(randomization)
    return image_info

def extract_info_from_filename(file):
    patientBD = None
    date_part = 0
    bd = None
    initials = None
    
    # Convert the filename to lowercase for case-insensitive comparison
    lowercase_filename = file.lower()
    if "." in lowercase_filename:
        lowercase_filename = lowercase_filename.rsplit(".", 1)[0]

    lowercase_filename = lowercase_filename.replace("-", "_")
    sub = re.sub(r'[!?,().]', "", lowercase_filename)
    stripped = sub.replace(" ", "_")
    # Check the first part for a combination of letters and numbers
    match = re.match(r'([a-zA-Z]+)(\d+)(\d+)', stripped)
    if match:
        letters, numbers, y = match.groups()
        stripped = f"{letters} {numbers} {y}"
        stripped = stripped.replace(" ", "_")
        
    parts = list(filter(None, stripped.split("_")))

    
    for part in parts:
        if len(part) == 6 and re.match(r'^\d+$', part):
            # If the part is exactly 6 digits, assume it's the birthdate
            bd = part
            break
    if bd:     
        bd_index = parts.index(bd)
        initials = parts[bd_index -1]
    
        if len(parts) > bd_index + 1:
            date_part_candidate = parts[bd_index + 1]
            if date_part_candidate.isdigit() and len(date_part_candidate) == 1:
                date_part = int(date_part_candidate)
            else: 
                date_part = 0
            
        patientBD = initials + " " + bd

    return patientBD, date_part


def sort_and_rename_images(input_dir, output_dir, info_file):
    image_info = load_image_info(info_file)
    counts = {}
    other_dir = os.path.join(output_dir, 'Other')
    os.makedirs(other_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                patient_folder, year, number = extract_info_from_filename(file)

                if patient_folder:
                    dates = image_info.get(patient_folder, []) if patient_folder else []
                    
                    if dates:
                        # Extract the year from the image filename
                        if year == "0":
                            year = file.lower().split(" ")[-2]
                        else:
                            year = year.split(" ")[0]

                        if (patient_folder, year) not in counts:
                            counts[(patient_folder, year)] = 0
                        else:
                            counts[(patient_folder, year)] += 1

                        new_name = f"{patient_folder} year {year} num {number}"

                        old_path = os.path.join(root, file)
                        new_path = os.path.join(output_dir, new_name + ".jpg")
                        shutil.copy2(old_path, new_path)
                    else:
                        other_path = os.path.join(other_dir, file)
                        shutil.copy2(os.path.join(root, file), other_path)
                        print(f"Warning: No valid dates for patient folder {patient_folder} in {file}")
                else:
                    other_path = os.path.join(other_dir, file)
                    shutil.copy2(os.path.join(root, file), other_path)

    print("Sorting and renaming completed.")



# def extract_info_from_filename(file):
#     """
#     Extract randomization number and date information from the image filename.
#     Splits the name and extracts the info accordingly.
    
#     Parameters:
#     - file (str): Image filename.

#     Returns:
#     - tuple: Randomization number and date.
#     """
#     # Convert the filename to lowercase for case-insensitive comparison
#     lowercase_filename = file.lower()
#     if "." in lowercase_filename:
#         lowercase_filename = lowercase_filename.rsplit(".", 1)[0]
    
#     # Replace any "-" with "_"
#     cleaned_filename = lowercase_filename.replace(" ", "_")
#     c_cleaned_filename = cleaned_filename.replace(",", "_")
    
#     # Split based on space, underscore, and comma
#     parts = c_cleaned_filename.split('_')

#     # Check if the file name structure is as expected
#     randomization_number = None
#     date_part = None

#     for part in parts:
#         if part.isdigit() and len(part) == 6:
#             # If a part is a digit, assume it's the randomization number
#             randomization_number = part
#             break

#     # If randomization number is found, extract the date
#     if randomization_number:
#         # Find the index of the randomization number in the parts list
#         index = parts.index(randomization_number)
        
#         # Extract the date from the part after the randomization number
#         if index + 1 < len(parts):
#             date_part = parts[index + 1][:6]

#     return randomization_number, date_part

def sort_and_rename_images(input_dir, output_dir, info_file):
    """
    Sorts and renames images in the specified directory based on information from the text file.

    Parameters:
    - input_dir (str): Path to the input directory containing the images.
    - output_dir (str): Path to the output directory for the sorted and renamed images.
    - info_file (str): Path to the text file containing randomization number and date information.

    Returns:
    None
    """
    image_info = load_image_info(info_file)
    counts = {}
    randomization_number = None

    # Create the 'Other' directory if it doesn't exist
    other_dir = os.path.join(output_dir, 'Other')
    os.makedirs(other_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                # Attempt to extract information from the image name
                patientBD, date = extract_info_from_filename(file)
                pbd = patientBD.upper() if patientBD else " "
                # Get the list of dates for the randomization number
                infor = image_info.get(pbd, []) if pbd else []
            
                # Check if there are dates for the randomization number
                if infor:
                    # Use the first date to extract the year
                    year = date if date is None else 0
                    randomization_number = infor[0] 
                    # Create the new file name
                    if (randomization_number, year) not in counts:
                        counts[(randomization_number, year)] = 0
                    else:
                        counts[(randomization_number, year)] += 1

                    new_name = f"{randomization_number} year {counts[(randomization_number, year)]}"

                    # Construct the full paths
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(output_dir, new_name + ".jpg")

                    # Move and rename the file
                    shutil.copy2(old_path, new_path)
                else:
                    # Move the file to the 'Other' directory
                    other_path = os.path.join(other_dir, file)
                    shutil.copy2(os.path.join(root, file), other_path)
                    print(f"Warning: No valid dates for Patient{patientBD} With RN {randomization_number} in {file}")
            else:
                # Move the file to the 'Other' directory
                other_path = os.path.join(other_dir, file)
                shutil.copy2(os.path.join(root, file), other_path)

    print("Sorting and renaming completed.")

# # Example usage:
# input_directory = "C:/Users/student/Desktop/images/processed_images/Aarhus_processed/armsDown"
# #input_directory = "C:/Users/student/Desktop/images/Sorted_images/Aarhus_sorted/armsDown/Other"
# output_directory = "C:/Users/student/Desktop/images/Sorted_images/Aarhus_sorted/armsDown"
# info_file = "C:/Users/student/Desktop/images/Aarhus.txt"  # CSV file
# sort_and_rename_images(input_directory, output_directory, info_file)
# Example usage:
# input_directory = "C:/Users/student/Desktop/images/processed_images/Aarhus_processed/armsUp"
# output_directory = "C:/Users/student/Desktop/images/Sorted_images/Aarhus_sorted/armsUp"
# info_file = "C:/Users/student/Desktop/images/Aarhus.txt"  # CSV file
# sort_and_rename_images(input_directory, output_directory, info_file)
# Example usage:
input_directory = "C:/Users/student/Desktop/images/processed_images/Aarhus_processed/Side"
output_directory = "C:/Users/student/Desktop/images/Sorted_images/Aarhus_sorted/Side"
info_file = "C:/Users/student/Desktop/images/Aarhus.txt"  # CSV file
sort_and_rename_images(input_directory, output_directory, info_file)

