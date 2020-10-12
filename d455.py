import pyrealsense2 as rs
import numpy as np
import cv2
import datetime

pipeline = rs.pipeline()
config = rs.config()

fps = 30
width = 1280
height = 720

#depth camera
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
#rgb camera
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
#ir camera
config.enable_stream(rs.stream.infrared, width, height, rs.format.y8, fps)

process = pipeline.start(config)

#設定Emitter_Enable模式
device =process.get_device()
depth_sensor = device.first_depth_sensor()
depth_sensor.set_option(rs.option.emitter_enabled, 0.0)

#saved file setting
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
today = datetime.datetime.now()
timestr = today.strftime("%Y%m%d-%H%M%S")

rgb_out = cv2.VideoWriter(timestr + '_rgb'+'.mp4',fourcc, fps, (width,height))
ir_out = cv2.VideoWriter(timestr + '_ir'+'.mp4',fourcc, fps, (width,height))
depth_out = cv2.VideoWriter(timestr + '_depth'+'.mp4',fourcc, fps, (width,height))

try:
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame()
        #ir_frame = frames.get_infrared_frame(1)#數字1代表左影像，數字2代表右影像，不加也可以
        #ir_frame = frames.get_infrared_frame(2)#數字1代表左影像，數字2代表右影像，不加也可以

        if not depth_frame or not color_frame or not ir_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        
        #print(color_image.shape)
        #print(ir_image.shape)
        
        ir_three_channel = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
        
        rgb_out.write(color_image)
        ir_out.write(ir_three_channel)
        depth_out.write(depth_colormap)

        cv2.imshow('RGB', color_image)
        cv2.imshow('IR', ir_image) 
        cv2.imshow('depth', depth_colormap)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


finally:

    # Stop streaming
    pipeline.stop()
    rgb_out.release()
    ir_out.release()
    depth_out.release()
