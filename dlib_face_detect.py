import os
import sys
import cv2 
import numpy as np
import CNN as cnn
import time
import utils
import dlib
from threading import Thread
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *

class VideoInput():
    box_left = 150
    box_right = 150
    box_up = 50
    box_down = 100
    def start_capture(self, model, fileUrl, mode):
        cap = cv2.VideoCapture(fileUrl)
        w = 0
        h = 0
        past_x1 = 400
        past_y1 = 400
        past_x2 = 500
        past_y2 = 500
        ret, frame = cap.read()
        height, width = frame.shape[:2]
        self.predicted_class = 0
        output = ""
        font = cv2.FONT_HERSHEY_SIMPLEX
        detector = dlib.get_frontal_face_detector()
        if(mode == 2):    
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(fps)
            print(width)
            print(height)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('./result.avi', fourcc, fps, (width, height))
        
        n=0
        while(cap.isOpened()):
            filename = "image_" + str(n) + ".jpg"
            output = ""
            n=n+1
            ret, frame = cap.read()
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
            mark_mat = frame.copy()
            face_rects = detector(frame, 0)
            #print(n)
            if(len(face_rects)>=1):
                #print(face_rects[0])
                #[(x1,y1),(x2,y2)] = face_rects[0]
                x1 = face_rects[0].left()
                y1 = face_rects[0].top()
                x2 = face_rects[0].right()
                y2 = face_rects[0].bottom()

                w=x2-x1
                h=y2-y1
                #print(x1 - past_x1)
                if( abs(x1-past_x1) < 150 or n<80): 
                    self.box_left = int((2*w + self.box_left)/2)
                    self.box_right = int((3.5*w + self.box_right)/2)
                    self.box_up = int((h/1.5 + self.box_up)/2)
                    self.box_down = int((height-y2 + self.box_down)/2)
                
                    past_x1 = int( (x1 + past_x1)/2 )
                    past_y1 = int( (y1 + past_y1)/2 )
                    past_x2 = int( (x2 + past_x2)/2 )
                    past_y2 = int( (y2 + past_y2)/2 )
                else:
                    x1 = past_x1
                    y1 = past_y1
                    x2 = past_x2
                    y2 = past_y2
                
            elif(len(face_rects)<1):
                x1 = past_x1
                y1 = past_y1
                x2 = past_x2
                y2 = past_y2


            if(w>50 and h>50):
                (x1_body, y1_body, x2_body, y2_body) = (x1-self.box_left, y1-self.box_up, x2+self.box_right, y2+self.box_down)
                if(x1_body<0):
                    x1_body=0
                if(y1_body<0):
                    y1_body=0
                if(x2_body>height):
                    x2_body=height
                if(y2_body>width):
                    y2_body=width

                out = frame[y1_body:y2_body,x1_body:x2_body]
                out = cv2.resize(out, (256, 256) )
                
                cv2.imshow("detect image", out) 
                
                output, model_out = self.predict_output(out , model)
                self.predicted_class = np.argmax(model_out)
                cv2.rectangle(mark_mat, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
                cv2.rectangle(mark_mat, (x1_body,y1_body),(x2_body,y2_body), (255, 0, 0), 4, cv2.LINE_AA)
                cv2.rectangle(mark_mat, (0, 0), (500, 100), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(mark_mat,output,(20,70), font, 1.5,(0,0,255),3,cv2.LINE_AA)

                if(mode == 1):
                    cv2.imwrite(filename, out)

            if(mode == 2):
                cv2.imwrite(filename, mark_mat)
                video_writer.write(mark_mat)
            
            cv2.imshow("Face Detection", mark_mat)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if(mode == 2):
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()


    def predict_output(self , image , model):
        image = cv2.resize(image, (utils.IMG_SIZE, utils.IMG_SIZE))
        image = np.array(image)
        data = image.reshape(utils.IMG_SIZE, utils.IMG_SIZE, 3)
        model_out = model.predict([data])
        return str(utils.IMAGE_CLASS[np.argmax(model_out)]),model_out

class Qt(QWidget):
    def mv_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "/home/hongze/Desktop","Mp4 (*.mp4)", options=opt)
	
        return fileUrl[0]


if __name__ == '__main__':
    if(len(sys.argv) < 2):
        mode =0
    elif(sys.argv[1] == '--convert'):
        mode = 1
    elif(sys.argv[1] == '--result'):
        mode = 2

    network = cnn.CNN()
    model = network.load_model()
    video = VideoInput()
    qt_env = QApplication(sys.argv)
    process = Qt()
    mv_fileUrl = process.mv_Chooser()
    print(mv_fileUrl)
    video.start_capture(model, mv_fileUrl, mode)
