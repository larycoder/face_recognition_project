import numpy as np
import cv2 as cv
#push image to memory for processing
def ImagePush():
    path=input('link of image: ')
    if path=="":
        path='../../image/face_2.jpg'
    img=cv.imread(path,0)
    return img

#show image
def ImageShow(img):
    cv.imshow('image',img)
    k=cv.waitKey()
    while k!=ord('q'):
        k=cv.waitKey()

#import module face using haar
def ImageDetect(img):
    Module=input('link trained module: ')
    if Module=='':
      Module='../../haarcascades/haarcascade_frontalface_default.xml'
    face_haar=cv.CascadeClassifier(Module)
    face=face_haar.detectMultiScale(img,1.1,5)
    print("Number of face is: ",len(face))
    print("local of point is: ",face)
    for(x,y,w,h) in face:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#run program
def ImageDetected():
    img=ImagePush()
    ImageDetect(img)
    ImageShow(img)
    return img

ImageDetected()