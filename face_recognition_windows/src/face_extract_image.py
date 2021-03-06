from face_detected_haar import ImageDetect
import cv2
import os

path='../../image/training-data/s2_Quoc-Viet'

#create folder for indentify
try:
    if not os.path.isdir(path):
        os.makedirs(path)
except OSError:
    pass
#load video to video object
video=cv2.VideoCapture("../../image/KieuQuocViet.mp4")

#count frames of video
video_length=int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#extract image to path
count,length=(0,0)
for i in range(20):
    for i in range(video_length//20):
        ret,frame=video.read()
    # convert image to gray and extract only face
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=ImageDetect(gray, 1)
    if len(face)==1:
        (x,y,w,h)=face[0]
        cv2.imwrite(path+"/%#05d.jpg"%(count+1),gray[y:y+w,x:x+h])
        count+=1
