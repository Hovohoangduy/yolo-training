import os
import shutil
directory = 'val/images'
for filename in os.listdir(directory):
    if filename.endswith(('.jpg','.JPG','.png','.jpeg')):
        with open('val.txt', 'a') as file:
            file.write('./val/images/'+ filename + '\n')