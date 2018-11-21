from face_train_LBPH import LBPH_face_trained
from face_detected_haar import*
import cv2

path='../../image/KieuQuocViet.mp4'
label_name=['unknow','Hung','Quoc Viet']


video=cv2.VideoCapture(path)
k=cv2.waitKey(1)
count=60
label_predict=(0,0)
while video.isOpened():
    ret,frame=video.read()
    face_matrix= ImageDetect(frame,1)
    if not len(face_matrix)==0:
        (x,y,w,h)=face_matrix[0]
        if count>=60:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            label_predict=LBPH_face_trained().predict(gray[y:y+w,x:x+h])
            print(label_predict)
            count=0
        count+=1
        face_draw_rectangle(frame,face_matrix)
        face_draw_text(frame,face_matrix,label_name[label_predict[0]])
    cv2.imshow('image',frame)
    k=cv2.waitKey(1)
    if k==27:
        break
video.release()
cv2.destroyAllWindows()
