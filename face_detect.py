import sys
import cv2 
import numpy as np
import CNN as cnn
import queue
import time
import utils
from threading import Thread
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *

IMG_SIZE = 128
image_class = ['Drinking','Talking Left ','Talking Right','Adjust Radio/Music Player','Texting Left ','Texting Right','Safe']


##----------------------------------------------Camera----------------------------------------------##

class Camera():
    def start_capture(self, model):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        self.predicted_class = 0
        output = ""
        font = cv2.FONT_HERSHEY_SIMPLEX
        while(True):
            start = time.time()
            ret, frame = cap.read()
            src = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if(len(faces)>=1):
                (x,y,w,h) = faces[0]
                center = (x + w//2, y + h//2)
                (x1_body, y1_body, x2_body, y2_body) = (x-int(1.2*w), y-int(h/5), x+int(2*w), y+3*h)
                face_mat = src[y1_body:y2_body,x1_body:x2_body]
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                frame = cv2.rectangle(frame,(x1_body,y1_body),(x2_body,y2_body),(0,255,0),2)

                output, model_out = self.predict_output(face_mat , model)
                self.predicted_class = np.argmax(model_out)
                end = time.time()
                seconds = end - start
                #print( "Time taken : {0} seconds".format(seconds))
                fps = "FPS : " + str(int(1 / seconds))
                #print( "Estimated frames per second : {0}".format(fps))
                cv2.putText(frame,fps,(10,50), font, 1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(frame,output,(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def predict_output(self , image , model):
        image = cv2.resize(image, (utils.IMG_SIZE, utils.IMG_SIZE))
        image = np.array(image)
        data = image.reshape(utils.IMG_SIZE, utils.IMG_SIZE, 3)
        model_out = model.predict([data])
        return str(utils.IMAGE_CLASS[np.argmax(model_out)]),model_out


##----------------------------------------------Video ----------------------------------------------##

class VideoInput():
    def start_capture(self , model, fileUrl):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(fileUrl)
        self.predicted_class = 0
        output = ""
        font = cv2.FONT_HERSHEY_SIMPLEX
        ret, frame = cap.read()
        while(cap.isOpened()):
            start = time.time()
            ret, frame = cap.read()
            src = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if(len(faces)>=1):
                (x,y,w,h) = faces[0]
                center = (x + w//2, y + h//2)
                (x1_body, y1_body, x2_body, y2_body) = (x-int(1.2*w), y-int(h/5), x+int(2*w), y+3*h)
                face_mat = src[y1_body:y2_body,x1_body:x2_body]
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                frame = cv2.rectangle(frame,(x1_body,y1_body),(x2_body,y2_body),(0,255,0),2)

                output, model_out = self.predict_output(face_mat , model)
                self.predicted_class = np.argmax(model_out)
                end = time.time()
                seconds = end - start
                #print( "Time taken : {0} seconds".format(seconds))
                fps = "FPS : " + str(int(1 / seconds))
                #print( "Estimated frames per second : {0}".format(fps))
                cv2.putText(frame,fps,(10,50), font, 1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(frame,output,(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def predict_output(self , image , model):
        image = cv2.resize(image, (utils.IMG_SIZE, utils.IMG_SIZE))
        image = np.array(image)
        data = image.reshape(utils.IMG_SIZE, utils.IMG_SIZE, 3)
        model_out = model.predict([data])
        return str(utils.IMAGE_CLASS[np.argmax(model_out)]),model_out

##----------------------------------------------  QT  ----------------------------------------------##

class Qt(QWidget):
    def mv_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "/home/hongze/Desktop","Mp4 (*.mp4)", options=opt)
	
        return fileUrl[0]

##---------------------------------------------- Main ----------------------------------------------##

if __name__ == '__main__':
    network = cnn.CNN()
    model = network.load_model()
    
    if(sys.argv[1] == '--camera'):
        cam = Camera()
        cam.start_capture(model)

    elif(sys.argv[1] == '--video'):
        video = VideoInput()
        qt_env = QApplication(sys.argv)
        process = Qt()
        mv_fileUrl = process.mv_Chooser()
        print(mv_fileUrl)
        video.start_capture(model, mv_fileUrl)
