import cv2 
import numpy as np 

#%% read the image into python env #%%

# convert the RGB image to grayscale
img1 = cv2.imread(r"D:\Docs and stuff\Profile Details\mask\Capture47.PNG")
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# apply corresponding threshold to convert to BW.
thresh, grayBW = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)

# Extract ROI.
imgSubSec = gray[804:825, 237:325]

# Apply same threshold as applied to the main image to convert the
# mask to BW
ret, mask = cv2.threshold(imgSubSec, 115, 255, cv2.THRESH_BINARY)
# perform bitwise NOT boolean operation to the mask image
invMask = cv2.bitwise_not(mask)

# create a blank image and insert the mask in the ROI
imgCnst = np.ones([913, 538], dtype = np.uint8) * 255
imgCnst[804:825, 237:325] = mask

# subtract the mask from the Main BW image to mask the ROI.
dst = cv2.subtract(imgCnst,grayBW)

# perform bitwise NOT to get the BW image.
reinv = cv2.bitwise_not(dst)
cv2.imshow('reinv', reinv)

#%% Mask second image

img2 = cv2.imread(r"D:\Docs and stuff\Profile Details\mask\Image_2.jpeg")
gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

thresh, grayBW1 = cv2.threshold(gray1, 100, 255, cv2.THRESH_BINARY)

imgSubSec1 = gray1[323:347, 357:514]

ret, mask1 = cv2.threshold(imgSubSec1, 100, 255, cv2.THRESH_BINARY)
invMask1 = cv2.bitwise_not(mask1)

imgCnst1 = np.ones([527, 924], dtype = np.uint8) * 255
imgCnst1[323:347, 357:514] = mask1

dst1 = cv2.subtract(imgCnst1, grayBW1)

reinv1 = cv2.bitwise_not(dst1)
cv2.imshow('reinv1', reinv1)

#%% Mask third image

img3 = cv2.imread(r"D:\Docs and stuff\Profile Details\mask\Image_3.jpeg")
gray2 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

thresh, grayBW2 = cv2.threshold(gray2, 100, 255, cv2.THRESH_BINARY)
imgSubSec2 = gray2[538:570, 163:312]
imgSubSec3 = gray2[815:839, 175:280]

ret, mask2 = cv2.threshold(imgSubSec2, 100, 255, cv2.THRESH_BINARY)
ret, mask3 = cv2.threshold(imgSubSec3, 100, 255, cv2.THRESH_BINARY)
invMask2 = cv2.bitwise_not(mask2)
invMask3 = cv2.bitwise_not(mask3)

imgCnst2 = np.ones([927, 505], dtype = np.uint8) * 255
imgCnst2[538:570, 163:312] = mask2
imgCnst2[815:839, 175:280] = mask3

dst2 = cv2.subtract(imgCnst2, grayBW2)

reinv2 = cv2.bitwise_not(dst2)

cv2.imshow('reinv2', reinv2) 

#%% EOL
