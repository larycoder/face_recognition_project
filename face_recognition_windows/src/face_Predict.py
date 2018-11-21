from face_train_LBPH import LBPH_face_trained
from face_detected_haar import*
import cv2

path='../../image/Hung.mp4'
label_name=['unknow','Hung','Quoc Viet']


video=cv2.VideoCapture(path)
k=cv2.waitKey(1)
count=0
label_predict=(0,0)
while video.isOpened():
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_matrix= ImageDetect(frame,1)
    count+=1
    if count>20:
        label_predict=LBPH_face_trained().predict(gray)
        print(label_predict)
        count=0
    face_draw_text(frame,face_matrix,label_name[label_predict[0]])
    cv2.imshow('image',frame)
    k=cv2.waitKey(1)
    if k==27:
        video.release()
        break
cv2.destroyAllWindows()
