import sys
import cv2 
import numpy as np
import time
import utils
import dlib
from threading import Thread
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *

img_size = 256

class VideoInput():
    box_left = 0
    box_right = 0
    box_up = 0
    box_down = 0
    init_x1=0
    init_x2=0
    init_y1=0
    init_y2=0
    def set_box(self, fileUrl):
        num = 0
        text = " Initialize ...... "
        font = cv2.FONT_HERSHEY_SIMPLEX
        cap = cv2.VideoCapture(fileUrl)
        ret, frame = cap.read()
        width, height = frame.shape[:2]
        detector = dlib.get_frontal_face_detector()
        while(cap.isOpened()):
            num = num + 1
            ret, frame = cap.read()
            mark_mat = frame.copy()
            face_rects = detector(frame, 1)
            if(len(face_rects)>=1):
                x1 = face_rects[0].left()
                y1 = face_rects[0].top()
                x2 = face_rects[0].right()
                y2 = face_rects[0].bottom()
            
                
                self.init_x1 = int((x1+self.init_x1)/2)
                self.init_x2 = int((x2+self.init_x2)/2)
                self.init_y1 = int((y1+self.init_y1)/2)
                self.init_y2 = int((y2+self.init_y2)/2)

                w = self.init_x2 - self.init_x1
                h = self.init_y2 - self.init_y1
                
                self.box_left = int((2*w + self.box_left)/2)
                self.box_right = int((3.5*w + self.box_right)/2)
                self.box_up = int((h/1.5 + self.box_up)/2)
                self.box_down = int((height-y2 + self.box_down)/2)

                (x1_body, y1_body, x2_body, y2_body) = (self.init_x1-self.box_left, self.init_y1-self.box_up, self.init_x2+self.box_right, self.init_y2+self.box_down)
                
                cv2.rectangle(mark_mat, (self.init_x1, self.init_y1), (self.init_x2, self.init_y2), (0, 255, 0), 4, cv2.LINE_AA)
                cv2.rectangle(mark_mat, (x1_body,y1_body),(x2_body,y2_body), (255, 0, 0), 4, cv2.LINE_AA)
            
            cv2.putText(mark_mat,text,(50,120), font, 4,(0,0,255),3,cv2.LINE_AA)
            cv2.imshow("Face Detection", mark_mat)

            if cv2.waitKey(1) & 0xFF == ord('q') :
                break
            if(num>60):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    def start_capture(self, fileUrl):
        cap = cv2.VideoCapture(fileUrl)
        past_x1 = self.init_x1
        past_y1 = self.init_y1
        past_x2 = self.init_x2
        past_y2 = self.init_y2
        
        ret, frame = cap.read()
        width, height = frame.shape[:2]
        detector = dlib.get_frontal_face_detector()
        n=0
        while(cap.isOpened()):
            filename = "pic_" + str(n) + ".jpg"
            n=n+1
            ret, frame = cap.read()
            mark_mat = frame.copy()
            face_rects = detector(frame, 0)
            if(len(face_rects)>=1):
                #print(face_rects[0])
                #[(x1,y1),(x2,y2)] = face_rects[0]
                x1 = face_rects[0].left()
                y1 = face_rects[0].top()
                x2 = face_rects[0].right()
                y2 = face_rects[0].bottom()
                
                past_x1 = int( (x1 + past_x1)/2 )
                past_y1 = int( (y1 + past_y1)/2 )
                past_x2 = int( (x2 + past_x2)/2 )
                past_y2 = int( (y2 + past_y2)/2 )

            elif(len(face_rects)<1):
                x1 = past_x1
                y1 = past_y1
                x2 = past_x2
                y2 = past_y2

            w=x2-x1
            h=y2-y1
            
            if(w>50 and h>50):
                (x1_body, y1_body, x2_body, y2_body) = (x1-self.box_left, y1-self.box_up, x2+self.box_right, y2+self.box_down)
                if(x1_body <0):
                    x1_body = 0
                #print(y1_body)
                #print(y2_body)
                #print(x1_body)
                #print(x2_body)
                out = frame[y1_body:y2_body,x1_body:x2_body]
                out = cv2.resize(out, (img_size, img_size) )
                
                cv2.rectangle(mark_mat, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
                cv2.rectangle(mark_mat, (x1_body,y1_body),(x2_body,y2_body), (255, 0, 0), 4, cv2.LINE_AA)
                cv2.imwrite(filename, out)
            
            cv2.imshow("Face Detection", mark_mat)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

class Qt(QWidget):
    def mv_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "/home/hongze/Desktop","Mp4 (*.mp4)", options=opt)
	
        return fileUrl[0]


if __name__ == '__main__':
    video = VideoInput()
    qt_env = QApplication(sys.argv)
    process = Qt()
    mv_fileUrl = process.mv_Chooser()
    print(mv_fileUrl)
    video.set_box(mv_fileUrl)
    video.start_capture(mv_fileUrl)
