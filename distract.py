import time
import cv2

import os
import sys
import numpy as np

import CNN as cnn
import time
import utils
import dlib
import openpyxl

from head_pose_estimation.mark_detector import MarkDetector
from head_pose_estimation.pose_estimator import PoseEstimator
from head_pose_estimation.stabilizer import Stabilizer


from threading import Thread
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
from gaze_tracking import GazeTracking
CNN_INPUT_SIZE = 128

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
        
        mark_detector = MarkDetector()
        pose_estimator = PoseEstimator(img_size=(height, width))
        pose_stabilizers = [Stabilizer(state_num=2,measure_num=1,cov_process=0.1,cov_measure=0.1) for _ in range(6)]
        tm = cv2.TickMeter()
        
        self.predicted_class = 0
        output = ""
        font = cv2.FONT_HERSHEY_SIMPLEX
        #臉部偵測
        #detector = dlib.get_frontal_face_detector()
        #predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        #gaze = GazeTracking()
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
            facebox = mark_detector.extract_cnn_facebox(frame)
            #創一個用來畫辨識集果的圖片
            start = time.time()
            mark_mat = frame.copy()
            #臉部偵測
            #face_rects = detector(frame, 0)
            cv2.rectangle(mark_mat, (0, 0), (width, 100), (255, 255, 255), -1, cv2.LINE_AA)
            mid = time.time()
            print(mid-start)
           #一個以上的臉部偵測,取第一個當作對象
            if(facebox is not None):
                face_img = frame[facebox[1]: facebox[3], facebox[0]: facebox[2]]
                cv2.imshow("face", face_img) 
                face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                marks = mark_detector.detect_marks([face_img]) 
                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]
                pose = pose_estimator.solve_pose_by_68_points(marks)
                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])
                steady_pose = np.reshape(steady_pose, (-1, 3))
                point2D = pose_estimator.draw_annotation_box(mark_mat, steady_pose[0], steady_pose[1], color=(128, 255, 128))
               #print(face_rects[0])
                x1 = facebox[0]
                y1 = facebox[1]
                x2 = facebox[2]
                y2 = facebox[3]
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
            elif(facebox is None):
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
                cv2.rectangle(mark_mat, (x1, y1), (x2, y2), (255, 255, 255), 4, cv2.LINE_AA)
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

