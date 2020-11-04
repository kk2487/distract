import time
import cv2
import numpy as np

from head_pose_estimation.mark_detector import MarkDetector
from head_pose_estimation.pose_estimator import PoseEstimator
from head_pose_estimation.stabilizer import Stabilizer

CNN_INPUT_SIZE = 128

fileUrl = "/home/hongze/distract/20201104-140913_rgb.mp4"

def main():
    
    cap = cv2.VideoCapture(fileUrl)
    _, sample_frame = cap.read()
    mark_detector = MarkDetector()
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('./result.avi', fourcc, 25, (width, height))
    
    while True:
        # Read frame, crop it, flip it, suits your needs.
        start = time.time()
        frame_got, frame = cap.read()
        if frame_got is False:
            break
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
        facebox = mark_detector.extract_cnn_facebox(frame)
        if facebox is not None:
            face_img = frame[facebox[1]: facebox[3], facebox[0]: facebox[2]]
            
            cv2.imshow("face", face_img) 
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            cv2.imshow("face_resize", face_img) 
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
            point2D = pose_estimator.draw_annotation_box(frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))
            #下層
            #左上
            cv2.circle(frame, tuple(point2D[0]), 4, (255,0,255), 8)
            #左下
            cv2.circle(frame, tuple(point2D[1]), 4, (255,0,255), 8)
            #右下
            cv2.circle(frame, tuple(point2D[2]), 4, (255,0,255), 8)
            #右上
            cv2.circle(frame, tuple(point2D[3]), 4, (255,0,255), 8)
            #上層
            #左上
            cv2.circle(frame, tuple(point2D[5]), 4, (255,255,0), 8)
            #左下
            cv2.circle(frame, tuple(point2D[6]), 4, (255,255,0), 8)
            #右下 
            cv2.circle(frame, tuple(point2D[7]), 4, (255,255,0), 8)
            #右上
            cv2.circle(frame, tuple(point2D[8]), 4, (255,255,0), 8)
            #cv2.imshow("face", face_img) 
        end = time.time()
        seconds = end - start

        #print("FPS : ", int(1/seconds))
        cv2.imshow("Preview", frame)
        #video_writer.write(frame)
        if cv2.waitKey(10) == 27:
            cap.release()
            cv2.destroyAllWindows()
            video_writer.release()
            break

if __name__ == '__main__':
    main()
