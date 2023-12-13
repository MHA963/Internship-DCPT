import pandas as pd
import re
import os
import shutil
import pickle

image_df = pd.DataFrame(columns=["Original Image Name", "Original Path", "Output Path", "Randomization Number", "Year", "Position"])

def load_image_info(file_path, hospital_structure):
    """
    Load file name, randomization number, and image year from a CSV file with explicit labels.

    Parameters:
    - file_path (str): Path to the CSV file.
    - hospital_structure (str): A string specifying the hospital structure (e.g., 'Aalborg', 'Vejle', 'Odense', 'Dresden', etc.).

    Returns:
    - dict: Dictionary mapping file names to tuples (randomization number, image year).
    """
    
    image_info = {}
    df = pd.read_excel(file_path)
    if hospital_structure == 'Aarhus': 
        for index, row in df.iterrows():
            patientBD= row['Patient folder']
            repeated_path = row['repeated path']
            randomization = row['Randomization number']
            image_date = row['Image date']
            year = row['Year']
            
            if not pd.isna(repeated_path) and not pd.isna(randomization) and not pd.isna(year):
                file_name = os.path.basename(repeated_path)
                image_info[patientBD] = (randomization, image_date)
        return image_info
    else: 
        for index, row in df.iterrows():
            repeated_path = row['repeated path']
            randomization = row['Randomization number']
            image_date = row['Image date']
            year = row['Year']

            if not pd.isna(repeated_path) and not pd.isna(randomization) and not pd.isna(year):
                file_name = os.path.basename(repeated_path)
                image_info[randomization] = (randomization, image_date)
        return image_info


def extract_info_from_filename(file, hospital_structure):
    """
    Extract randomization number and date information from the image filename based on the hospital structure.

    Parameters:
    - file (str): Image filename.
    - hospital_structure (str): A string specifying the hospital structure (e.g., 'Aalborg', 'Vejle', 'Odense', 'Dresden', etc.).

    Returns:
    - tuple: Randomization number and date.
    """
    # Convert the filename to lowercase for case-insensitive comparison
    lowercase_filename = file.lower()
    if "." in lowercase_filename:
        lowercase_filename = lowercase_filename.rsplit(".", 1)[0]
    
    randomization_number = None
    date_part = None
    
    if hospital_structure == 'Aalborg':
        # Extract information specific to Aalborg structure
        cleaned_filename = lowercase_filename.replace("hypo-", "").replace("hypo_", "")
        cleaned_filename = cleaned_filename.replace("-", "_")
        parts = cleaned_filename.split('_')
        
        if len(parts) >= 3:
            randomization_number = parts[0]
            date_part = parts[1][:6] if len(parts) > 2 else ""
            return randomization_number, date_part
    
    elif hospital_structure == 'Aarhus': 
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
    
    
    elif hospital_structure == 'Vejle':
        # Split the file name into words
        
            
        #parts = c_cleaned_filename.split('_')
        # Replace any "-" with "_"
        cleaned_filename = lowercase_filename.replace(" ", "_")
        c_cleaned_filename = cleaned_filename.replace(", ", "_")
        c_cleaned_filename = cleaned_filename.replace(",","")
        # Split based on space, underscore, and comma
        parts = c_cleaned_filename.split('_')

        # Check if the file name structure is as expected

        for part in parts:
            if part.isdigit() and len(part) == 6:
                # If a part is a digit, assume it's the randomization number
                randomization_number = part
                break

        # If randomization number is found, extract the date
        if randomization_number:
            # Find the index of the randomization number in the parts list
            index = parts.index(randomization_number)
            
            # Extract the date from the part after the randomization number
            if index + 1 < len(parts):
                date_part = parts[index + 1][:6]

        return randomization_number, date_part

    elif hospital_structure == 'Odense':
        # Extract information specific to Odense structure
        cleaned_filename = lowercase_filename.replace(" ", "_")
        parts =  cleaned_filename.split('_')
        
        if len(parts) >= 3:
            randomization_number = parts[0]
            #extract the date from the fourth part of the filename.
            if any(char.isdigit() for char in parts[1]):
                date_part = parts[1][:6]
            else:
                date_part = parts[2][:6] if len(parts) > 3 else ""
            
            return randomization_number, date_part

    elif hospital_structure == 'Dresden':
        # Extract information specific to Dresden structure
        if "." in lowercase_filename:
            lowercase_filename = lowercase_filename.rsplit(".", 1)[0]
        
        cleaned_filename = lowercase_filename.replace(",", "_")
        c_cleaned_filename = cleaned_filename.replace(" ", "_")
        # Split based on space, underscore, and comma
        parts = c_cleaned_filename.split('_')
        
        for part in parts:
            if part.isdigit() and len(part) == 6:
                # If a part is a digit, assume it's the randomization number
                randomization_number = part
                break

        # If randomization number is found, extract the year from the part after the randomization number
        if randomization_number:
            index = parts.index(randomization_number)
            if index + 1 < len(parts):
                # Remove any non-numeric characters and consider the result as the year
                date_part = ''.join(filter(str.isdigit, parts[index + 1]))
                if date_part:
                    year = int(date_part)
                else:
                    # If there's no part after the randomization number, assume the year is 0
                    year = 0
            
            return randomization_number, year

    else:
        # Handle other hospital structures or unknown structures here...
        pass

    # If none of the above conditions matched, return None for both values.
    return None, None


def sort_and_rename_images(input_dir, output_dir, info_file, hospital_structure):
    """
    Sorts and renames images in the specified directory based on information from the text file for a given hospital structure.

    Parameters:
    - input_dir (str): Path to the input directory containing the images.
    - output_dir (str): Path to the output directory for the sorted and renamed images.
    - info_file (str): Path to the text file containing randomization number and date information.
    - hospital_structure (str): A string specifying the hospital structure (e.g., 'Aalborg', 'Vejle', 'Odense', 'Dresden', etc.).

    Returns:
    None
    """
    image_info = load_image_info(info_file,hospital_structure)
    counts = {}
    processed_images =  []
    # Create the 'Other' directory if it doesn't exist
    other_dir = os.path.join(output_dir, 'Other')
    os.makedirs(other_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                # Attempt to extract information from the image name based on the hospital structur
                if hospital_structure in ('Aalborg', 'Vejle'):
                    randomization_number, date = extract_info_from_filename(file, hospital_structure)
                
                    dates = image_info.get(randomization_number, []) if randomization_number else []
                    image_name = os.path.basename(file).lower()
                    if randomization_number and date:
                        randomization_number = int(randomization_number)
                        date = int(date)
                        # Get the list of dates for the randomization number
                    else: 
                        randomization_number = 0
                        date = 0
                
                    if dates and randomization_number !=0 and date:
                        # Find the image info based on the image name
                        #image_info_for_image = image_info.get(image_name+ ' ', None)
                        year = dates[1].year  
                    # Create the new file name
                        if (randomization_number, year) not in counts:
                            counts[(randomization_number, year)] = 0
                        else:
                            counts[(randomization_number, year)] += 1

                        new_name = f"{randomization_number} year {counts[(randomization_number, year)]}"

                        #new_name = f"{randomization_number} year {year}"

                        # Construct the full paths
                        old_path = os.path.join(root, file)
                        new_path = os.path.join(output_dir, new_name + ".jpg")

                        # Move and rename the file
                        shutil.copy2(old_path, new_path)
                        processed_images.append([file, os.path.abspath(os.path.join(root, file)), os.path.abspath(new_path), randomization_number, date, os.path.basename(output_dir)])

                    else:
                        # Move the file to the 'Other' directory
                        other_path = os.path.join(other_dir, file)
                        shutil.copy2(os.path.join(root, file), other_path)
                        print(f"Warning: No valid dates for randomization number {randomization_number} in {file}")
                
                elif hospital_structure == 'Aarhus':
                    patientBD, date = extract_info_from_filename(file,hospital_structure)
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
                        processed_images.append([file, os.path.abspath(os.path.join(root, file)), os.path.abspath(new_path), randomization_number, date, os.path.basename(output_dir)])

                    else:
                        # Move the file to the 'Other' directory
                        other_path = os.path.join(other_dir, file)
                        shutil.copy2(os.path.join(root, file), other_path)
                        print(f"Warning: No valid dates for Patient{patientBD} in {file}")
                    
                elif hospital_structure == 'Odense':
                        randomization_number, date = extract_info_from_filename(file, hospital_structure)
                
                        dates = image_info.get(randomization_number, []) if randomization_number else []
                        image_name = os.path.basename(file).lower()
                        if randomization_number and date:
                            randomization_number = int(randomization_number)
                            date = int(date)
                            # Get the list of dates for the randomization number
                        else: 
                            randomization_number = 0
                            date = 0
                            
                        if dates and randomization_number !=0 and date:
                            # Find the image info based on the image name
                            #image_info_for_image = image_info.get(image_name+ ' ', None)
                            year = dates[1].year  
                        # Create the new file name
                            if (randomization_number, year) not in counts:
                                counts[(randomization_number, year)] = 0
                            else:
                                counts[(randomization_number, year)] += 1

                            new_name = f"{randomization_number} year {counts[(randomization_number, year)]}"

                            #new_name = f"{randomization_number} year {year}"

                            # Construct the full paths
                            old_path = os.path.join(root, file)
                            new_path = os.path.join(output_dir, new_name + ".jpg")

                            # Move and rename the file
                            shutil.copy2(old_path, new_path)
                            processed_images.append([file, os.path.abspath(os.path.join(root, file)), os.path.abspath(new_path), randomization_number, date, os.path.basename(output_dir)])

                        else:
                            # Move the file to the 'Other' directory
                            other_path = os.path.join(other_dir, file)
                            shutil.copy2(os.path.join(root, file), other_path)
                            print(f"Warning: No valid dates for randomization number {randomization_number} in {file}")
                    
                elif hospital_structure == 'Dresden':
                    randomization_number, date = extract_info_from_filename(file, hospital_structure)
                
                    dates = image_info.get(randomization_number, []) if randomization_number else []
                    image_name = os.path.basename(file).lower()
                    if randomization_number and date:
                        randomization_number = int(randomization_number)
                        date = int(date)
                        # Get the list of dates for the randomization number
                    else: 
                        randomization_number = 0
                        date = 0
                    if randomization_number is not None and date is not None:
                        new_name = f"{randomization_number} year {date}"

                        # Construct the full paths
                        old_path = os.path.join(root, file)
                        new_path = os.path.join(output_dir, new_name + ".jpg")

                        # Move and rename the file
                        shutil.copy2(old_path, new_path)
                        processed_images.append([file, os.path.abspath(os.path.join(root, file)), os.path.abspath(new_path), randomization_number, date, os.path.basename(output_dir)])

                    else:
                        # Move the file to the 'Other' directory
                        other_path = os.path.join(other_dir, file)
                        shutil.copy2(os.path.join(root, file), other_path)
                        print(f"Warning: File {file} doesn't match the provided information.")
                          
            else:
                # Move the file to the 'Other' directory
                other_path = os.path.join(other_dir, file)
                print("Something is wrong, maybe naming.")
                shutil.copy2(os.path.join(root, file), other_path)
    
    save_to_excel(processed_images, output_dir + '/output_images.xlsx')
    print("Sorting and renaming completed.")


def save_to_excel(processed_images, outputfile):
    """
    Save processed image information to an Excel file.

    Parameters:
    - processed_images (list): List of processed image data.
    - outputfile (str): Path to the output Excel file.

    Returns:
    None
    """
    image_df = pd.DataFrame(processed_images, columns=["Original Image Name", "Original Path", "Output Path", "Randomization Number", "Year", "Position"])
    image_df.to_excel(outputfile,index= False)