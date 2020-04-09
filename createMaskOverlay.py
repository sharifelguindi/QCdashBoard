from skimage import data, color, io, img_as_float
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
import json
from textwrap import dedent as d
from dateutil.relativedelta import relativedelta
from datetime import datetime

import pandas as pd
from skimage import io, data
import numpy as np
import cv2
import PIL
import scipy.io as sio
from skimage import exposure

alpha = 0.6

img = img_as_float(data.camera())
rows, cols = img.shape

# Construct a colour image to superimpose
color_mask = np.zeros((rows, cols, 3))
color_mask[30:140, 30:140] = [1, 0, 0]  # Red block
color_mask[170:270, 40:120] = [0, 1, 0] # Green block
color_mask[200:350, 200:350] = [0, 0, 1] # Blue block

# Construct RGB version of grey-level image
img_color = np.dstack((img, img, img))

# Convert the input image and color mask to Hue Saturation Value (HSV)
# colorspace
img_hsv = color.rgb2hsv(img_color)
color_mask_hsv = color.rgb2hsv(color_mask)

# Replace the hue and saturation of the original image
# with that of the color mask
img_hsv[..., 0] = color_mask_hsv[..., 0]
img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

img_masked = color.hsv2rgb(img_hsv)

# Display the output
f, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                  subplot_kw={'xticks': [], 'yticks': []})
ax0.imshow(img, cmap=plt.cm.gray)
ax1.imshow(color_mask)
ax2.imshow(img_masked)
plt.show()

## Helper functions for datetime parsing ##
epoch = datetime.utcfromtimestamp(0)
def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0

def get_marks_from_start_end(start, end):
    ''' Returns dict with one item per month
    {1440080188.1900003: '2015-08',
    '''
    result = []
    current = start
    while current <= end:
        result.append(current)
        current += relativedelta(months=1)
    return {unix_time_millis(m)/100000:(str(m.strftime('%Y-%m'))) for m in result}

## Initial DB Pull ##
df = pd.read_excel("D:\\pythonProjects\\autoQC\\headneckDB.xls")
i = 0
for dtObj in df['Datetime']:
    if isinstance(dtObj,str):
        try:
            dtObjConverted = datetime.strptime(dtObj, '%Y%m%d %H%M%S.%f')
            df['Datetime'][i] = dtObjConverted
        except ValueError:
            dtObjConverted = datetime.strptime(dtObj, '%Y%m%d %H%M%S')
            df['Datetime'][i] = dtObjConverted
    i = i + 1
df.index = df['Datetime']
filtered_col = [col for col in df if col.endswith('DCE')]
df_DCE_only = df[filtered_col]
marksDict = get_marks_from_start_end(df['Datetime'].min(), df['Datetime'].max())

dataFile = df.iloc[1]['basePath'] + os.sep + df.iloc[1]['PlanCFileName']
pixels = sio.loadmat(dataFile)
volume = pixels['data'][1, 9]
bbox = pixels['data'][1,12]
bbox = bbox[0,:]
height = width = 128
canvas_width = 500
canvas_height = round(height * canvas_width / width)
scale = canvas_width / width

planCFile = dataFile.replace('data','planC')
planCStruct = sio.loadmat(planCFile)
scan = planCStruct['planC'][0,2][0,0]['scanArray'].astype('uint16')
scan = scan
mask = np.zeros(np.shape(scan))
# plt.imshow(scan[:,:,100],cmap='gray');plt.show()
mask[bbox[0]:bbox[1]+1,bbox[2]:bbox[3]+1,bbox[4]:bbox[5]+1] = volume
centroid_contour = [int(np.mean([bbox[0],bbox[1]+1])), int(np.mean([bbox[2],bbox[3]+1])), int(np.mean([bbox[4],bbox[5]+1]))]

mask_img = mask[centroid_contour[0]-int(height/2):centroid_contour[0]+int(height/2),
                centroid_contour[1]-int(height/2):centroid_contour[1]+int(height/2),
                centroid_contour[2]]
scan_img = scan[centroid_contour[0]-int(height/2):centroid_contour[0]+int(height/2),
                centroid_contour[1]-int(height/2):centroid_contour[1]+int(height/2),
                centroid_contour[2]]


# scan_img = np.clip(scan_img, 900, 1150)
scan_img_norm = np.zeros(np.shape(scan_img),dtype='uint8')
scan_img_norm = cv2.normalize(scan_img, scan_img_norm,0 ,255,cv2.NORM_MINMAX)
alpha = 0.5
blended_img = alpha * mask_img*255 + (1 - alpha) * scan_img_norm