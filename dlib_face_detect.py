import os
import sys
import cv2 
import numpy as np
import CNN as cnn
import time
import utils
import dlib
import openpyxl

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
        wb =openpyxl.Workbook()
        sheet = wb['Sheet']

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
        
        n=1
        while(cap.isOpened()):
            
            filename = "image1_" + str(n) + ".jpg"
            index = 'A'+str(n)
            output = ""
            n=n+1
            #print(index)
            ret, frame = cap.read()
            #順時鐘旋轉90度
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
            #創一個用來畫辨識集果的圖片
            start = time.time()
            mark_mat = frame.copy()
            #臉部偵測
            face_rects = detector(frame, 0)
            cv2.rectangle(mark_mat, (0, 0), (width, 100), (255, 255, 255), -1, cv2.LINE_AA)
            mid = time.time()
            print(mid-start)
            
            """
            #偵測瞳孔位置 
            gaze.refresh(frame)
            #print(gaze.horizontal_ratio())
            #print(gaze.vertical_ratio())
            #print("------------------")
            #水平
            text = ""
            if ( gaze.horizontal_ratio() != None): 
                if (float(gaze.horizontal_ratio())<0.6) :
                    text = "right"
                elif (float(gaze.horizontal_ratio())>=0.8):
                    text = "left"
                else :
                    text = "center"
            cv2.putText(mark_mat, text, (350, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            #垂直
            text = ""
            if ( gaze.vertical_ratio() != None): 
                if (float(gaze.vertical_ratio())>0.9) :
                    text = "down"
                elif(float(gaze.vertical_ratio())<0.1) :
                    text = "up"
                else:
                    text = "center"
                
            cv2.putText(mark_mat, text, (350, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            """
           #一個以上的臉部偵測,取第一個當作對象
            if(len(face_rects)>=1):
                """
                #偵測頭部歪斜,轉頭
                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame,face_rects[0]).parts()])
                
                eye_l_x = int((landmarks[42,0] + landmarks[45,0])/2)
                eye_l_y = int((landmarks[42,1] + landmarks[45,1])/2)
                eye_r_x = int((landmarks[36,0] + landmarks[39,0])/2)
                eye_r_y = int((landmarks[36,1] + landmarks[39,1])/2)
                
                m_eye = ((height-eye_l_y)-(height-eye_r_y))/(eye_l_x-eye_r_x)
                dis_eye = pow(eye_l_x-eye_r_x,2) + pow(eye_l_y-eye_r_y,2)
                
                dis = pow(landmarks[0,0]-landmarks[36,0],2)+pow(landmarks[0,1]-landmarks[36,1],2)
                print("%.3f" %m_eye)
                print(dis_eye)
                print(dis)
                print("-----------------------")
                text = ""
                if(m_eye>0.1):
                    text = "tilt right"
                elif(m_eye<-0.2):
                    text = "tilt left"
                else:
                    text = "normal"
                cv2.putText(mark_mat, text, (550, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                #print(m_h+m_eye)
                text = ""
                if(dis<3000):
                    text = "turn right"
                elif(dis>10000):
                    text = "turn left"
                else:
                    text = "normal"
                cv2.putText(mark_mat, text, (550, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                #cv2.line(mark_mat, (landmarks[27,0],landmarks[27,1]), (landmarks[30,0],landmarks[30,1]), (0,0,255), 2)
                cv2.line(mark_mat, (0,landmarks[27,1]),(width,landmarks[27,1]), (0,0,0), 2) 
                #cv2.line(mark_mat, (xx1,yy1),(xx2,yy2), (255,0,0), 2) 
                cv2.line(mark_mat, (eye_l_x,eye_l_y),(eye_r_x,eye_r_y), (0,255,0), 5) 
                for idx, point in enumerate(landmarks):        
                    #enumerate函式遍歷序列中的元素及它們的下標
                    #68點的座標
                    pos = (point[0, 0], point[0, 1])
                    #print(idx,pos)
                    cv2.circle(mark_mat, pos, 2, color=(150, 0, 0))

                """
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
                
                #cv2.imshow("detect image", out) 
                
                output, model_out = self.predict_output(out , model)
                self.predicted_class = np.argmax(model_out)
                cv2.rectangle(mark_mat, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
                cv2.rectangle(mark_mat, (x1_body,y1_body),(x2_body,y2_body), (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(mark_mat,output,(20,50), font, 1.5,(0,0,255),3,cv2.LINE_AA)

                #sheet[index] = output
                #wb.save('result.xlsx')
                #儲存訓練圖片
                #if(mode == 1):
                #    cv2.imwrite(filename, out)
            #儲存結果圖與結果影片
            #if(mode == 2):
            #    cv2.imwrite(filename, mark_mat)
            #    video_writer.write(mark_mat)
            end = time.time()
            seconds = end - mid

            print("FPS : ", int(1/seconds))
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
