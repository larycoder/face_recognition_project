from face_train_LBPH import LBPH_face_trained
from face_detected_haar import*
import cv2

path='../../image/video.mp4'
video=cv2.VideoCapture(path)
ret,frame=video.read()
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
label_predict=LBPH_face_trained().predict(gray)
print(label_predict)
