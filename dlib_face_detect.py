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
from gaze_tracking import GazeTracking

class VideoInput():
    #臉部範圍與辨識位置的差量：往左'往右'往上'往下
    box_left = 150
    box_right = 150
    box_up = 50
    box_down = 200
    def start_capture(self, model, fileUrl, mode):
        cap = cv2.VideoCapture(fileUrl)
        w = 0
        h = 0
        #紀錄上一個辨識範圍
        past_x1 = 400
        past_y1 = 400
        past_x2 = 500
        past_y2 = 500
        ret, frame = cap.read()
        #順時鐘轉90度
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
        height, width = frame.shape[:2]
        self.predicted_class = 0
        output = ""
        font = cv2.FONT_HERSHEY_SIMPLEX
        #臉部偵測
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        gaze = GazeTracking()
        #儲存結果圖片、影片
        if(mode == 2):    
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(fps)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('./result.avi', fourcc, fps, (width, height))
        
        n=0
        while(cap.isOpened()):
            filename = "image1_" + str(n) + ".jpg"
            output = ""
            n=n+1
            ret, frame = cap.read()
            #順時鐘旋轉90度
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
            #創一個用來畫辨識集果的圖片
            mark_mat = frame.copy()
            #臉部偵測
            face_rects = detector(frame, 0)
            
            
            gaze.refresh(frame)
            print(gaze.horizontal_ratio())
            print(gaze.vertical_ratio())
            print("------------------")
            text = ""
            cv2.rectangle(mark_mat, (0, 150), (300, 250), (255, 255, 255), -1, cv2.LINE_AA)
            if ( gaze.horizontal_ratio() != None): 
                if (float(gaze.horizontal_ratio())<0.6) :
                    text = "Looking right"
                elif (float(gaze.horizontal_ratio())>=0.8):
                    text = "Looking left"
                else :
                    text = "Looking center"
            cv2.putText(mark_mat, text, (20, 180), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            text = ""
            if ( gaze.vertical_ratio() != None): 
                if (float(gaze.vertical_ratio())>0.9) :
                    text = "Looking down"
                elif(float(gaze.vertical_ratio())<0.1) :
                    text = "Looking up"
                else:
                    text = "Looking center"
                
            cv2.putText(mark_mat, text, (20, 230), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            
            #一個以上的臉部偵測,取第一個當作對象
            if(len(face_rects)>=1):
                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame,face_rects[0]).parts()])
                x1 = landmarks[27,0]
                y1 = landmarks[27,1]
                x2 = landmarks[30,0]
                y2 = landmarks[30,1]
                m = ((height - y2) -(height-y1)) / (x2 - x1 + 0.000000001)
                m_h = -1 / m
                #print(m_vertical)
                xx1 = 200
                xx2 = 600
                yy1 = int(height - m_h * xx1 -(height - y2))
                yy2 = int(height - m_h * xx2 -(height - y2))
                
                m_eye = ((height-landmarks[39,1])-(height-landmarks[42,1]))/(landmarks[39,0]-landmarks[42,0])

                print(m_eye, m_h)
                
                xe1 = 200
                xe2 = 600
                ye1 = int(height - m_eye * xx1 -(height - landmarks[39,1]))
                ye2 = int(height - m_eye * xx2 -(height - landmarks[39,1]))
                
                cv2.line(mark_mat, (landmarks[27,0],landmarks[27,1]), (landmarks[30,0],landmarks[30,1]), (0,0,255), 2)
                cv2.line(mark_mat, (xx1,yy1),(xx2,yy2), (255,0,0), 2) 
                cv2.line(mark_mat, (xe1,ye1),(xe2,ye2), (0,255,0), 5) 
                for idx, point in enumerate(landmarks):        
                    #enumerate函式遍歷序列中的元素及它們的下標
                    #68點的座標
                    pos = (point[0, 0], point[0, 1])
                    #print(idx,pos)
                    cv2.circle(mark_mat, pos, 2, color=(150, 0, 0))

               #print(face_rects[0])
                x1 = face_rects[0].left()
                y1 = face_rects[0].top()
                x2 = face_rects[0].right()
                y2 = face_rects[0].bottom()
                #計算臉部寬高
                w=x2-x1
                h=y2-y1

                #當與上一個frame的臉部位置差距合理內,或程式剛執行,計算臉部位置與辨識位置差量,並更新past point
                if( abs(x1-past_x1) < 150 or n<80):
                    #當前與原本取平均
                    self.box_left = int((x1-0 + self.box_left)/2)
                    self.box_right = int((width-x2 + self.box_right)/2)
                    self.box_up = int((h + self.box_up)/2)
                    self.box_down = int((height-y2 + self.box_down)/2)

                    past_x1 = int( (x1 + past_x1)/2 )
                    past_y1 = int( (y1 + past_y1)/2 )
                    past_x2 = int( (x2 + past_x2)/2 )
                    past_y2 = int( (y2 + past_y2)/2 )
                else:
                    #若與上一個frame距離差過多,採用上一個frame紀錄的點
                    x1 = past_x1
                    y1 = past_y1
                    x2 = past_x2
                    y2 = past_y2

            #如果沒有偵測到臉部,採用上一個frame紀錄的點
            elif(len(face_rects)<1):
                x1 = past_x1
                y1 = past_y1
                x2 = past_x2
                y2 = past_y2

            #當寬高大於一定大小,開始計算預辨識範圍
            if(w>50 and h>50):
                (x1_body, y1_body, x2_body, y2_body) = (x1-self.box_left, y1-self.box_up, x2+self.box_right, y2+self.box_down)
                #超過邊界,例外處理
                if(x1_body<0):
                    x1_body=0
                if(y1_body<0):
                    y1_body=0
                if(x2_body>width):
                    x2_body=width
                if(y2_body>height):
                    y2_body=height
                #裁切預辨識範圍
                out = frame[y1_body:y2_body,x1_body:x2_body]
                #縮放成cnn輸入圖片大小
                out = cv2.resize(out, (256, 256) )
                
                cv2.imshow("detect image", out) 
                
                output, model_out = self.predict_output(out , model)
                self.predicted_class = np.argmax(model_out)
                cv2.rectangle(mark_mat, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
                cv2.rectangle(mark_mat, (x1_body,y1_body),(x2_body,y2_body), (255, 0, 0), 4, cv2.LINE_AA)
                cv2.rectangle(mark_mat, (0, 0), (500, 100), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(mark_mat,output,(20,70), font, 1.5,(0,0,255),3,cv2.LINE_AA)

                #儲存訓練圖片
                if(mode == 1):
                    cv2.imwrite(filename, out)
            #儲存結果圖與結果影片
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
