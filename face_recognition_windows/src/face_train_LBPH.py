import cv2
import numpy
import os


path='../../image/training-data'

#prepare data to train
def prepare_training_data(train_folder=path):
    dirs=os.listdir(train_folder)
    faces=[]
    labels=[]
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label=int(dir_name[1])
        image_dir=train_folder+'/'+dir_name
        images=os.listdir(image_dir)
        for image_name in images:
            if image_name.startswith("."):
                continue
            image=cv2.imread(image_dir+'/'+image_name)
            faces.append(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
            labels.append(label)
    return faces,labels



#training data to LBPH face recognition
def LBPH_face_trained(train_folder=path):
    faces, labels = prepare_training_data(train_folder)
    # print total faces and labels:
    ##print('total faces: ', len(faces))
    ##print('total labels: ', len(labels))
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,numpy.array(labels))
    return face_recognizer
