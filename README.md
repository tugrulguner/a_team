TEAM: SaferWherever

The aim of this project is to develop a solution based on the use of AI for a post COVID-19 world as part of the #BuildwithAI : Emergence! Hackathon.

easierfaster.py contains all necessary density and mask detection algorithms in it and return density population, total amount of mask and no-mask detected through live camera selected from https://www.insecam.org/ showing a good spot in Colorado, USA. It is less accurate compared to YOLO_main one but faster.

YOLO_main.py uses YOLO algorithm to detect faces, and it counts total number of people in that way. Then it calculates total number of mask on and mask off people together with total number of social distance violations as outputs, and upload data to web app. You can download YOLO model weights from https://pjreddie.com/darknet/yolo/ for the model YOLOv3-416 or for any model you want. This is the model we are using for app, but if you want faster model, you can use easierfaster.py

You can check the app from here: https://a-team-mall-monitor.herokuapp.com/

It updates continously in time
