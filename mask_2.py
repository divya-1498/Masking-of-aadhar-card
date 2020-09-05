from imutils import contours 
import numpy as np
import imutils
import cv2

img1 = cv2.imread(r"D:\Docs and stuff\Profile Details\mask\Capture47.PNG")
img2 = cv2.imread(r"D:\Docs and stuff\Profile Details\mask\ocra1.png")



ref = cv2.imread(r"D:\Docs and stuff\Profile Details\mask\ocra1.png")
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, 
                           cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {} 

for(i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + x:x +w]
    roi = cv2.resize(roi, (57, 88))
    
    digits[i] = roi
    
    
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

image = cv2.imread(r"D:\Docs and stuff\Profile Details\mask\Capture47.PNG")
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
                  ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
#gradX = gradX.astype("unit8")

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 1, 255,
                       cv2.THRESH_BINARY , cv2.THRESH_OTSU)[0]

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)


