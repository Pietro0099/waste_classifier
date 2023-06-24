import os
import random

folder_path = "raw_data/train/plastic"  # Path to directory to shuffle and rename
extension = ".jpg"  # Replace with your file extension

# Get all image filenames in folder
files = [f for f in os.listdir(folder_path) if f.endswith(extension)]

# Shuffle the filenames
random.shuffle(files)

# Rename and save the files with a unique new name
for i, file_name in enumerate(files):
	new_file_name = f"a_{i+1:04}{extension}"  # Format the new filename as "a_0001.jpg", "a_0002.jpg", etc.
	os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
