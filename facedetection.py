import cv2

def faceDetectorImg(path):
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread(path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


    
