import cv2
#push image to memory for processing
def ImagePush():
    path=input('link of image: ')
    if path=="":
        path='../../image/face_2.jpg'
    img=cv2.imread(path,0)
    return img

#show image
def ImageShow(img):
    cv.imshow('image',img)
    k=cv.waitKey()
    while k!=ord('q'):
        k=cv.waitKey()

#import module face using haar
def ImageDetect(img,mode=0):
    if mode==0:
        Module=input('link trained module: ')
    else:
        Module=''
    if Module=='':
      Module='../../haarcascades/haarcascade_frontalface_default.xml'
    face_haar=cv2.CascadeClassifier(Module)
    face=face_haar.detectMultiScale(img,1.1,5)
    #print("Number of face is: ",len(face))
    #print("local of point is: ",face)
    if not mode==0:
        return face

#add text to image
def face_draw_text(img,matrix,text):
    (x,y,w,h)=matrix[0]
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,1.5,(0, 255, 0), 2)

#add rectangle to image
def face_draw_rectangle(img,face):
    for(x,y,w,h) in face:
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)