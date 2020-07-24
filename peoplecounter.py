def peoplecounter(path):
# import the necessary packages
# this code is modified based on the one provided in this link: https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
  from imutils.object_detection import non_max_suppression
  from imutils import paths
  import numpy as np
  import argparse
  import imutils
  import cv2
  import os
  from google.colab import drive
  drive.mount('/content/drive')
  
  hog = cv2.HOGDescriptor()
  hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
  image = cv2.imread(path)
  image = imutils.resize(image, width=min(400, image.shape[1]))
  orig = image.copy()
	# detect people in the image, we can adjust winStride and padding parameters to make it better
  (rects, weights) = hog.detectMultiScale(image, winStride=(2, 2),
	padding=(8, 8), scale=1.05)
	# draw the original bounding boxes
  return rects.shape[0]
