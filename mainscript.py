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

def faceDetectorImg(image):
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def MaskDetector(image, tolerance):
# I add tolerance add a new input here, which defines plus and minus pixel amount from the bounding box values.
# Idea is when we have bounding box that detects the face part of the body, it can draw rectangle covering the whole face
# but it can better to enlarge the covering area with the tolerance value. It justs enlarges the drawed box in both positive and negative directions.
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
          maskcounter += 1
        else:
          nomaskcounter += 1
    return [maskcounter, nomaskcounter]

### END ###

### Main BODY ###

def main():
  # Here we can define a single image for test purposes, if you want to check it, you can uncomment below
  #image = cv2.imread('/content/drive/My Drive/Colab_Notebooks/test.jpg')
  urlden = 'https://a-team-mall-api.herokuapp.com/density'
  urlmask = 'https://a-team-mall-api.herokuapp.com/mask'
 
  while True:
    # Here I provided a link from https://www.insecam.org/, which is a good spot from Colorado, USA
    cap = cv2.VideoCapture('http://208.139.200.133/mjpg/video.mjpg#.XxzWNVzDvVY.link')
    _, image = cap.read()

    facecount = faceDetectorImg(image)
    if len(facecount)==0:
        facecountstreaming = {"x": 70, 
                              "y": 20, 
                              "count": 0
                              }
        maskstreaming = {'x': 70,
                         'y': 20,
                        'mask': 0,
                        'nomask': 0
                         }
    else:
      facecount = facecount.shape[0]
      facecountstreaming = {"x": 70, 
                            "y": 20, 
                            "count": facecount
                            }
      # I used tolerance = 10 for the MaskDetector function, it can be adjusted	
      maskamount = MaskDetector(image, 10)
      maskstreaming = {'x': 70,
                       'y': 20,
                       'mask': maskamount[0],
                       'nomask': maskamount[1]
                       }
    # Here I am just trying to check whether I can post the data to our app
    gg = requests.post(urlden, data = facecountstreaming)
    zz = requests.post(urlmask, data = maskstreaming)
    if zz.status_code == requests.codes.ok:
      print('Mask Uploaded')
    # This print below is to show people count, amount of mask on and off manually on console, you can disable this part
    print('Density: ' + str(facecountstreaming['count']) 
    + ', Mask On: ' + str(maskstreaming['mask']) 
    + ', Mask Off: ' + str(maskstreaming['nomask']))
    k = cv2.waitKey(30) & 0xff
    if k==27:
      break
    # Release the VideoCapture object
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

