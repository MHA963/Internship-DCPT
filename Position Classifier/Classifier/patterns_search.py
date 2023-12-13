import os
import re

def find_file_naming_patterns(folder_path, patterns_to_match):
    matching_patterns = {}
    other_patterns = set()
    pattern_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in patterns_to_match]
    total_image_count = 0

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filename_lower = filename.lower()
            matched = False
            # Check if the lowercased file name matches any of the provided patterns
            for pattern_regex in pattern_regexes:
                if pattern_regex.match(filename_lower):
                    if pattern_regex.pattern in matching_patterns:
                        matching_patterns[pattern_regex.pattern] += 1
                    else:
                        matching_patterns[pattern_regex.pattern] = 1
                    matched = True
                    break
            # If the file name didn't match any of the provided patterns, store it as an "other" pattern
            if not matched:
                other_patterns.add(filename_lower)

            # Increment the total image count for all images
            if filename_lower.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                total_image_count += 1

    return matching_patterns, other_patterns, total_image_count

folder_path = "C:/Users/student/Desktop/images/processed_images/Vejle_processed/armsDown"

patterns_to_match = [
    #Add more patterns
    r'\d+ hypo [a-z]+ \d+_\d+_\d+\.jpg',
    r'hypo [a-z]+ \d+_\d+_\d+\.jpg', #5
    r'\d+ hypo [a-z]+ \d+, \d+\.jpg', #314
    r'\d+ hypo [a-z]+ \d+, \d+ .jpg', 
    r'\d+ hypo [a-z]+ \d+,\d+\.jpg', #4
    r'hypo [a-z]+ \d+, \d+\.jpg', #96
    
    
    
]

matching_patterns, other_patterns, total_images = find_file_naming_patterns(folder_path, patterns_to_match)

print("Matching Patterns:")
for pattern, count in matching_patterns.items():
    print(f"Pattern: {pattern}, Count: {count}")

print("Other Patterns:")
for pattern in other_patterns:
    print(f"Pattern: {pattern}")

print(f"Total Image Count: {total_images}")


#folder_path = "C:/Users/student/Desktop/images/processed_images/Vejle_processed/armsUp"
