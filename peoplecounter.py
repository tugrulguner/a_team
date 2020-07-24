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

### for visualization purposes ###
#for (x, y, w, h) in rects:
#	cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
#rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
#pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	# draw the final bounding boxes
#for (xA, yA, xB, yB) in pick:
#	cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	# show some information on the number of bounding boxes
#filename = imagePath[imagePath.rfind("/") + 1:]
#print("[INFO] {}: {} original boxes, {} after suppression".format(
#	filename, len(rects), len(pick)))
	# show the output images
#cv2_imshow(orig)
#cv2_imshow(image)
#cv2.waitKey(0)

###
