import cv2
#import numpy as np
test1=cv2.imread('/home/hiep/Desktop/github/face-recognition-project/image/face_2.jpg')
grayImg=cv2.cvtColor(test1,cv2.COLOR_BGR2GRAY)
haar_engine=cv2.CascadeClassifier('/home/hiep/Desktop/github/face-recognition-project/haarcascades/haarcascade_frontalface_default.xml')
face=haar_engine.detectMultiScale(grayImg,1.1,5)
print('face found: ',len(face))
for (x,y,w,h) in face:
    cv2.rectangle(test1,(x,y),(x+w,y+h),(0,255,0),2)
cv2.resizeWindow('image',300,600)
cv2.imshow('image',test1)
key=cv2.waitKey(0)
while(True):
    key=cv2.waitKey(0)
    if key==ord('s'):
        break
    else:
        continue
cv2.destroyAllWindows()
