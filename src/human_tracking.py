#! /usr/bin/env python3

import threading
import cv2
import numpy as np
from transforms3d.affines import compose
from transforms3d.euler import *
from transforms3d.quaternions import *
from pupil_apriltags import Detector
# from retinaface import RetinaFace
from ur5_intpro.msg import CamPose
import  rospy
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from zed_camera import ZED_CAMERA
import time

ZED_SERIAL_NUM = 33582849
    
class HUMAN_TRACKING(ZED_CAMERA):
    
    def __init__(self) -> None:
        ZED_CAMERA.__init__(self,serial_num=ZED_SERIAL_NUM,resolution="HD2K")
        rospy.init_node("human_tracking")
        self.cv_bridge = CvBridge()
        self.rgb_pub = rospy.Publisher("/zed/rgb",Image,queue_size=1)
        self.depth_pub = rospy.Publisher("/zed/depth",Image,queue_size=1)
        self.hpose_pub = rospy.Publisher("/human_pose",CamPose,queue_size=1)
        self.keypoints = list()
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640,640))
        self.at_detector = Detector(families="tag36h11",
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0
                                    )
        self.camparas = self.get_zed_params().get("params")
        self.origin_R = np.array([[ 0.78493768 , 0.58021624 ,-0.21730612],
                                  [-0.25926416 , 0.6261457  , 0.73533914],
                                  [ 0.56272101 ,-0.52085571 , 0.64191463]])
        self.origin_P = np.array([0.40878014,0.59730986,2.35316389])
        self.tf_origin = compose(self.origin_P, self.origin_R,[1,1,1])
        # self.tf_origin = compose(self.origin_P, np.eye(3),[1,1,1])
        self.moving_avg_buff = 10
        self.moving_avg_counter = 0
        self.filtered_xyz = np.zeros((self.moving_avg_buff,3),np.float32)
        self.filtered_x = []
    
    def moving_average(self,array, window_size = 6, get_mean_mvavg =True ):
        if len(array) <= window_size:
            print("need bigger array than window size or reduce window size")
            return array
        idx = 0
        array_mvavg = []
        while idx < (len(array)-window_size):
            window_avg = np.mean(array[idx:idx+window_size])
            array_mvavg.append(window_avg)
            idx += 1
            
        if get_mean_mvavg:
            array_mvavg = np.mean(array_mvavg)
            
        return array_mvavg
        
    def pub_msgs(self):
        frames = self.grab_zed_frame()
        rgb   = cv2.cvtColor(frames.get("rgb").get_data(),cv2.COLOR_RGBA2RGB)
        rgb   = self.cv_bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        depth = self.cv_bridge.cv2_to_imgmsg(frames.get("depth").get_data(),encoding="32FC1")
        self.rgb_pub.publish(rgb)
        self.depth_pub.publish(depth)
        
    
    def draw_keypoints(self, image, face) -> None:
        kps = face.kps.astype(int)
        for idx, l in enumerate(kps):
            color = (0,0,255)
            if idx == 0  or idx == 3:
                color = (0,255,0)
            cv2.circle(image,(l[0], l[1]), 1, color, 2)
            cv2.putText(image,f"{idx}",((l[0], l[1])), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    
    
    def draw_triangle(self, image, kps):
        kps = kps.astype(int)
        kps = np.delete(kps,2,axis=0)
        kps = np.asarray([kps[0],kps[1],kps[3],kps[2]])        
        cv2.fillConvexPoly(image,kps,(155,0,0))
    
    def draw_bbox(self,image, face) -> None:
        box = face.bbox.astype(int)
        color = (0, 0, 255)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        # if not (face.gender is None or face.age is None):
        #     cv2.putText(image,f"{face.sex},{face.age}", 
        #                 (box[0]-1, box[1]-4),
        #                 cv2.FONT_HERSHEY_COMPLEX,
        #                 0.7,(0,255,0),1)moving_avg_buff
    
    
    def tag_detector(self,rgb):
        gray = cv2.cvtColor(np.copy(rgb),cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray, 
                                       estimate_tag_pose=True, 
                                       camera_params=[self.camparas.fx,self.camparas.fy,self.camparas.cx,self.camparas.cy],
                                       tag_size=0.16)
        for tag in tags:
            print(tag)
    
    def show_frames(self) -> None:
        frames = self.grab_zed_frame(point_cloud=True)  
        rgb   = cv2.cvtColor(frames.get("rgb").get_data(),cv2.COLOR_RGBA2RGB)
        depth = frames.get("depth").get_data()
        point_cloud = frames.get("point_cloud")
        faces = self.app.get(rgb)

        for face_id, face in enumerate(faces):
            if face_id > 0:
                continue
            self.draw_bbox(rgb,face)
            if face.kps is not None:
                print("found face key points")
                self.draw_keypoints(rgb,face)
                kps = face.kps
                kp_depth = []
                for kp in kps:
                    try:
                        e, d = point_cloud.get_value(int(kp[0]),int(kp[1]))
                        # if e != "SUCCESS":
                        #     continue
                        kp_depth.append(d)
                        
                    except Exception as ex:
                        print(ex)                
            
                kp_depth = np.array(kp_depth)[:,:3]
                x_kp = kp_depth[:,0]
                x_kp = x_kp[np.logical_not(np.isnan(x_kp))]

                y_kp = kp_depth[:,1]
                y_kp = y_kp[np.logical_not(np.isnan(y_kp))]
                
                z_kp = kp_depth[:,2]
                z_kp = z_kp[np.logical_not(np.isnan(z_kp))]
                
                self.filtered_xyz[self.moving_avg_counter,:] = [np.mean(x_kp), np.mean(y_kp), np.mean(z_kp)]
                
                if self.moving_avg_counter == self.moving_avg_buff-1:
                    print("rolling")
                    self.filtered_xyz =np.roll(self.filtered_xyz,-1,axis=0)
                    
                    msg = CamPose()
                    # check if we keep z constant
                    test_y = np.sqrt(np.square(self.moving_average(self.filtered_xyz[:,2])) - (np.square(1.5)))
                    msg.x = self.moving_average(self.filtered_xyz[:,0])#self.filtered_xyz[0]
                    msg.y = 1.5 - test_y  #self.filtered_xyz[1]
                    # msg.y = self.moving_average(self.filtered_xyz[:,1])#self.filtered_xyz[1]
                    msg.z = self.moving_average(self.filtered_xyz[:,2])#self.filtered_xyz[2]
                    # self.hpose_pub.publish(msg)

                    if np.isnan(msg.x) or np.isnan(msg.y) or np.isnan(msg.z):
                        msg = ""
                        pass
                    else:
                        self.hpose_pub.publish(msg)
                        msg = f"face location (meters): x = {msg.x:03f}, y = {msg.y:03f}, z = {msg.z:03f}"
                    
                    cv2.putText(rgb,msg,(50,300), cv2.FONT_HERSHEY_PLAIN,  2, (0,255,0),2)
                    
                else:
                    self.moving_avg_counter += 1
                
        cv2.imshow("rgb",cv2.resize(rgb, (800,600)))
        cv2.waitKey(1)

    
        
if __name__ == '__main__':
    ht  = HUMAN_TRACKING()
    while not rospy.is_shutdown():
        try:
            t1 = time.time()
            ht.show_frames()
            print(f"obtain results at {1/(time.time()-t1):0.2f} hz")
            # ht.pub_msgs()
            rospy.sleep(1/100)
        except Exception as e:
            print(e)
        
        # t1 = time.time()
        # ht.show_frames()
        # print(f"obtain results at {1/(time.time()-t1):0.2f} hz")
        # rospy.sleep(1/100) 
        
        


# pose_R = [[ 0.78493768  0.58021624 -0.21730612]
#  [-0.25926416  0.6261457   0.73533914]
#  [ 0.56272101 -0.52085571  0.64191463]]
# pose_t = [[0.40878014]
#  [0.59730986]
#  [2.35316389]]
