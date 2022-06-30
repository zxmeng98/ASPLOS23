
import os
import pandas as pd
import glob


extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# Combine files
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.replace(['mobilenet_v3_small'], ['MobileNetV3'], inplace=True)

# Extract as csv
combined_csv.to_csv( "vgg.csv", index=False, encoding='utf-8-sig')
