
# import roslibpy
import numpy as np
import os
import sys
import rospy

from std_msgs.msg import Float64MultiArray, Float64
from geometry_msgs.msg import PoseStamped, Pose
from transforms3d.euler import *
from transforms3d.quaternions import *
from transforms3d.affines import *

from intpro_utils import *
import pyzed.sl as sl
import cv2
import argparse
from threading import Thread


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class SUB_DATA:
    
    def __init__(self,**kwargs) -> None:
        rospy.init_node("data_collection_node")
        self.package_path = os.path.dirname(os.path.abspath(os.path.join(__file__,"..")))
   
        self.general_params      = intpro_utils.load_yaml(os.path.join(self.package_path,"configs","general_params.yaml"))
        self.projector_params    = intpro_utils.load_proj_params(os.path.join(self.package_path,"configs",self.general_params["projector_calibration"]))
        self.resolution_codes = {"HD2K"  :{"camera": sl.RESOLUTION.HD2K  , "projector" : {"width": int(2048), "height": int(1080)} },
                                 "HD1080":{"camera": sl.RESOLUTION.HD1080, "projector" : {"width": int(1920), "height": int(1080)} },
                                 "HD720" :{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(1280), "height": int(720)}  },
                                 "VGA"   :{"camera": sl.RESOLUTION.VGA   , "projector" : {"width": int(640),  "height": int(320)}  },
                                #  "Custom":{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(4096), "height":  int(2160)}  }}
                                 "Custom":{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(1920), "height":  int(1080)}  }}
                                #  "Custom":{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(800), "height":  int(600)}  }}
                                
        self.tmp = list()      
        self.met_flag = True    
        self.met_data    = Float64MultiArray()
        self.ee_pose     = PoseStamped()
        self.lh_pose     = PoseStamped()
        self.rh_pose     = PoseStamped()
        self.pov_pose     = PoseStamped()
        
        self.zed_depth_map   = sl.Mat()
        self.zed_depth_image = sl.Mat()
        self.zed_rgb_image   = sl.Mat()
        
        
        self.frame_    = self.general_params["zed_frame"]
        self.zed       = sl.Camera()
    
        
        self.zed_init_params = sl.InitParameters()
        self.zed_init_params.set_from_serial_number(self.general_params["zed_serial"])
        self.zed_init_params.camera_resolution = self.resolution_codes[self.general_params["zed_res_dc"]]["camera"]
        self.zed_init_params.camera_fps = self.general_params["zed_fps"]
        if not self.zed.open(self.zed_init_params):
            print("Unable to open ZED camera")
            exit(1)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS,               self.general_params["zed_brightness"])
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST,                 self.general_params["zed_contrast"])
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE,                      self.general_params["zed_hue"])
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION,               self.general_params["zed_saturation"])
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS,                self.general_params["zed_sharpness"])
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN,                     self.general_params["zed_gain"])
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE,                 self.general_params["zed_exposure"])
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, self.general_params["zed_whitebal_temp"])
        self.zed_params = self.get_zed_params()
        
        
        
        rospy.Subscriber("/vrpn_client_node/EE/pose", PoseStamped,self.ee_pose_callback)
        rospy.Subscriber("/vrpn_client_node/HandLeft/pose", PoseStamped,self.lh_pose_callback)
        rospy.Subscriber("/vrpn_client_node/HandRight/pose", PoseStamped,self.rh_pose_callback)
        rospy.Subscriber("/vrpn_client_node/POV/pose", PoseStamped,self.pov_pose_callback)
        rospy.Subscriber("/epoc/met",Float64MultiArray,self.met_callback)
        self.zed_thread = Thread(target=self.zed_datacallback,args=())
        self.zed_thread.start()
    
    def get_zed_params(self) -> dict:
            params = self.zed.get_camera_information().camera_configuration.calibration_parameters
            if self.frame_ == "left":
                params = params.left_cam
            else:
                params = params.right_cam        
            intrinsic  = np.array([[params.fx, 0.0       , params.cx],
                                [0.0      , params.fy , params.cy],
                                [0.0      , 0.0       , 1.0]])
            
            distortion = np.array(params.disto)    
            return {"intrinsic": intrinsic, "distortion":distortion, "params": params}
        
    def ee_pose_callback(self,msg):
        self.ee_pose  = msg
    
    def lh_pose_callback(self,msg):
        self.lh_pose = msg
        
    def rh_pose_callback(self,msg):
        self.rh_pose = msg
    
    def pov_pose_callback(self,msg):
        self.pov_pose = msg
        
    def met_callback(self,msg):
        self.met_data = msg
        # if self.tmp == self.met_data.data:
        #     self.met_flag = False
        # else:
        #     self.met_flag = True
        # self.tmp = self.met_data.data
        
    def zed_datacallback(self):
        while not rospy.is_shutdown():
            try:
                if self.zed.grab(sl.RuntimeParameters()):
                    self.zed.retrieve_image(self.zed_rgb_image,sl.VIEW.LEFT if self.frame_=="left" else sl.VIEW.RIGHT)
                    self.zed.retrieve_measure(self.zed_depth_map,sl.MEASURE.DEPTH)
                    self.zed.retrieve_image(self.zed_depth_image, sl.VIEW.DEPTH)
                else:
                    print("!!!!!!Unable to grab frames from camera!!!!!!")
                rospy.sleep(1/30)
            except Exception as e:
                print(f"Zed callback error: {e}")
          
    def get_compose(self,pose_msg):        
        xyz = [pose_msg.pose.position.x,
               pose_msg.pose.position.y,
               pose_msg.pose.position.z]
        quat = [pose_msg.pose.orientation.w,
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z]
        return compose(xyz,quat2mat(quat),[1,1,1])
        
        
    def data_collect(self):
        while not rospy.is_shutdown():
            try:
                
                user_init = rospy.get_param("user_init")
                user_name = rospy.get_param("user_name")
                user_dir = rospy.get_param("user_dir")

                if not user_init:
                    continue
                
                # user_dir  = os.path.join("/share/icra_dataset","user_"+user_name)
                if not os.path.isdir(user_dir):
                    os.mkdir(user_dir)
                         
                current_mode = rospy.get_param("current_mode")
                if current_mode != "unknown":
                    user_dir = os.path.join(user_dir,current_mode)
                    if not os.path.isdir(user_dir):
                        os.mkdir(user_dir)
                else:
                    print("Current mode unknown....")
                    continue
                
                if not rospy.get_param("stop_saving"):
                    print(f"Saving data for user: {user_name} and mode: {current_mode}")
                    time_stamp = intpro_utils.get_timestamp()
                    np.save(os.path.join(user_dir,time_stamp+"_lf_pose.npy"),self.get_compose(pose_msg=self.lh_pose))
                    np.save(os.path.join(user_dir,time_stamp+"_rf_pose.npy"),self.get_compose(pose_msg=self.rh_pose))
                    np.save(os.path.join(user_dir,time_stamp+"_ee_pose.npy"),self.get_compose(pose_msg=self.ee_pose))
                    np.save(os.path.join(user_dir,time_stamp+"_pov_pose.npy"),self.get_compose(pose_msg=self.pov_pose))
                    np.save(os.path.join(user_dir,time_stamp+"_depth_map.npy"),self.zed_depth_map.get_data())   
                    np.save(os.path.join(user_dir,time_stamp+"_rgb_image.npy"),self.zed_rgb_image.get_data())
                    np.save(os.path.join(user_dir,time_stamp+"_timedelta.npy"),rospy.get_param("time_delta",default=1.0))
                    image = cv2.cvtColor(self.zed_rgb_image.get_data(),cv2.COLOR_BGRA2BGR)
                    h,w,c = image.shape
                    # cv2.imshow("image",cv2.resize(image, (int(w/2),int(h/2))))
                    # cv2.waitKey(1)
                    # np.save(os.path.join(user_dir,time_stamp+"_met_data.npy"),self.met_data.data)

                else:
                    print(f"Stopped saving data..... current user name: {user_name}")
                rospy.sleep(1/30)
            except KeyboardInterrupt:
                print("Shutting down data collection node!!!!")
                
                


if __name__ == "__main__":
    
    data_collect=  SUB_DATA()
    data_collect.data_collect()
    