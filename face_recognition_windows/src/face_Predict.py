from face_train_LBPH import LBPH_face_trained
from face_detected_haar import*
import cv2

def face_draw_text(img,matrix,text):
  for (x,y,w,h) in matrix:
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2)

path='../../image/video.mp4'
video=cv2.VideoCapture(path)
ret,frame=video.read()
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
face_matrix= ImageDetect(gray,1)
(x,y,w,h)=face_matrix[0]
face_compare=gray[y:y+w,x:x+h]
label_predict=LBPH_face_trained().predict(face_compare)
ImageDetect(frame)
face_draw_text(frame,face_matrix,label_predict)
cv2.imshow('image',frame)

