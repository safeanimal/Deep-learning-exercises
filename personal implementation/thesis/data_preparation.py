import os
import shutil

image_dirs = [
    'E:\datasets\BSD100_SR\image_SRF_2',
'E:\datasets\BSD100_SR\image_SRF_3',
    'E:\datasets\BSD100_SR\image_SRF_4'
]
for image_dir in image_dirs:
    os.chdir(image_dir)

    # Loop through all files in the directory
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.png'):  # Checks if the file is a PNG image
            # This splits the file name by '_' and takes the second last element as category
            category = file_name.split('_')[-1].split('.')[-2]

            # Create a new directory for the category if it doesn't exist
            if not os.path.exists(category):
                os.makedirs(category)

            # Move the file into its category directory
            shutil.move(file_name, os.path.join(category, file_name))

    print("Images have been categorized.")
