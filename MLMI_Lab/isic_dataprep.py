import pandas as pd
import os
import shutil

isic_gt = pd.read_csv('ISIC_2019_Training_GroundTruth.csv', sep = ',', encoding = 'unicode_escape')

labels_dict = {}              # labels dictionary 
count = 0
for key in isic_gt.keys():
  if (key != 'image'):
    labels_dict[key] = isic_gt[isic_gt[key]!=0]
    count += len(labels_dict[key])

assert count == len(isic_gt) 


basepath = 'ISIC_2019_Training_Input/'
for key in isic_gt.keys():
   if (key != 'image'):
      images = labels_dict[key].image.astype(str).unique() # Get all the images of particular label
      images = images + '.jpg'
      
      dest_path = os.path.join(basepath,key)
      if not os.path.isdir(dest_path):
        os.mkdir(dest_path)                               # Make a directory with class label name
      destination = os.path.join(basepath,key)

      for entry in os.listdir(basepath):
         if entry in images:
            filePath = os.path.join(basepath, entry)
            shutil.move(filePath, destination)              # move the image to particular label folder on matching
