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
    def start_capture(self, fileUrl):
        cap = cv2.VideoCapture(fileUrl)
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
                
                w=x2-x1
                h=y2-y1
                
                (x1_body, y1_body, x2_body, y2_body) = (x1-int(1.2*w), y1-int(h/2), x2+int(3*w), height)
                
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
    video.start_capture(mv_fileUrl)
