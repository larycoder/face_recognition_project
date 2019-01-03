from face_train_LBPH import LBPH_face_trained
from face_detected_haar import*
import cv2
import _thread

# predefined
path='../../image/Hung.mp4'
module='../../haarcascades/haarcascade_frontalface_default.xml'
label_name=['unknow','Hung','Quoc Viet']

#prepare tracker
tracker=cv2.TrackerBoosting_create()

# add video or stream image
video=cv2.VideoCapture(path)

# detect face
ret,frame=video.read()
face_matrix= ImageDetect(frame,module)
while len(face_matrix)==0:      # run detect until reach of 1 face
    ret,frame=video.read()
    face_matrix= ImageDetect(frame,module)

# track face
ok=tracker.init(frame,tuple(face_matrix[0]))

# set first label is unknow
label_predict=(0,0)

# run tracking
def face_tracking():
    while video.isOpened():
        # tracked and take matrix for drawing
        ret,frame=video.read()
        ok,face_tracked=tracker.update(frame)
        face_tracked=(int(face_tracked[0]),int(face_tracked[1]),int(face_tracked[2]),int(face_tracked[3]))
        #draw and notice
        if ok:
            # tracking successed
            face_draw_rectangle(frame,face_tracked)
            face_draw_text(frame,face_tracked,label_name[label_predict[0]])
        else:
            # tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        #show imge
        cv2.imshow('image',frame)
        k=cv2.waitKey(1)
        if k==27:
            break


        # take label recognition
        #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #(x,y,w,h)=face_matrix[0]
        #label_predict=LBPH_face_trained().predict(gray[y:y+w,x:x+h])
        #print(label_predict)
        # draw lable into image

try:
    _thread.start_new_thread(face_tracking(),())     
except:
    print("error unable to start thread\n")
video.release()
cv2.destroyAllWindows()
