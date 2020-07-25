###  Importing the Necessary Libraries ###

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
from tensorflow.keras.models import model_from_json
import requests
import json
from google.colab import drive
drive.mount('/content/drive')
import matplotlib.pyplot as plt

### END ###

### Function definitons ###

def peoplecounter(image):

# import the necessary packages
# this code is modified based on the one provided in this link: https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/

  hog = cv2.HOGDescriptor()
  hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
  image = imutils.resize(image, width=min(400, image.shape[1]))
  orig = image.copy()
	# detect people in the image, we can adjust winStride and padding parameters to make it better
  (rects, weights) = hog.detectMultiScale(image, winStride=(2, 2),
	padding=(8, 8), scale=1.05)
	# draw the original bounding boxes
  return rects.shape[0]

def faceDetectorImg(image):
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def MaskDetector(image, tolerance):

    def prepImg(img):
      return cv2.resize(img,(224,224)).reshape(1,224,224,3)/255.0
  
    with open('/content/drive/My Drive/Colab_Notebooks/DATA/model_mask.json', 'r') as f:
      loadthehason = f.read()
    model = model_from_json(loadthehason)

    model.load_weights("/content/drive/My Drive/Colab_Notebooks/DATA/model_mask.h5")
    faces = faceDetectorImg(image)
    maskcounter = 0
    nomaskcounter = 0
    for (x, y, w, h) in faces:
        slicedImg = image[(y-tolerance):(y+tolerance+h),(x-tolerance):(x+tolerance+w)]
        pred = model.predict(prepImg(slicedImg))
        pred = np.argmax(pred)
        if pred==0:
          mascounter += 1
        else:
          nomaskcounter += 1
    return [maskcounter, nomaskcounter]

### END ###

### Main BODY ###

def main():
  #image here is to check resulting values over any example image. You can uncomment that part if you want to check it over a single image
  #image = cv2.imread('/content/drive/My Drive/Colab_Notebooks/test.jpg')
  urlden = 'https://a-team-mall-api.herokuapp.com/density'
  urlmask = 'https://a-team-mall-api.herokuapp.com/mask'
  while True:
    #This part is essential, we can use fixed path for the camera or video.
    cap = cv2.VideoCapture('path')
    _, image = cap.read()
    #peopcount = peoplecounter(image)
    #densitystreaming = {'x': 70, 'y': 20, 'count': peopcount}
    #requests.post(urlden, data = densitystreaming)
    facecount = faceDetectorImg(image)
    facecount = facecount.shape[0]
    facecountstreaming = {"x": 70, 
                          "y": 20, 
                          "count": facecount
                          }
    gg = requests.post(urlden, data = facecountstreaming)
    if gg.status_code == requests.codes.ok:
      print('Density Uploaded')
    maskamount = MaskDetector(image, 10)
    maskstreaming = {'x': 70,
                     'y': 20,
                     'mask': maskamount[0],
                     'nomask': maskamount[1]
                     }
    zz = requests.post(urlmask, data = maskstreaming)
    if zz.status_code == requests.codes.ok:
      print('Mask Uploaded')
    print('Density: ' + str(facecount) + ', Mask On: ' + str(maskamount[0]) + ', Mask Off: ' + str(maskamount[1]))
    k = cv2.waitKey(30) & 0xff
    if k==27:
      break
    # Release the VideoCapture object
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
