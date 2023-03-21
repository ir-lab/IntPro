#! /usr/bin/en python3

import numba
from curses.ascii import isdigit
from cryptography.fernet import Fernet
import argparse
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys
import os
import numpy as np
from transforms3d.affines import compose
from transforms3d.euler import *
from transforms3d.quaternions import *
import cv2
from pupil_apriltags import Detector
import pyzed.sl as sl
from intpro_utils import *
import threading
from std_msgs.msg import Float64MultiArray, Float64, Float32MultiArray
import time
from tqdm import tqdm
from typing import *
import random
import traceback

class INTPRO:
    def __init__(self, **kwargs) -> None:
        rospy.init_node("intpro")
        
        ######################################################       
        # initialize projector and general parameters
        self.package_path        = os.path.dirname(os.path.abspath(os.path.join(__file__,"..")))
        self.home_tf             = np.load(os.path.join(self.package_path,"files","home_tf.npy"))
        self.projector_extrinsic = np.load(os.path.join(self.package_path,"files","projector_extrinsic.npy")) 
        self.projector_intrinsic = np.load(os.path.join(self.package_path,"files","projector_intrinsic.npy")) 
        self.general_params      = intpro_utils.load_yaml(os.path.join(self.package_path,"configs","general_params.yaml"))
        self.projector_params    = intpro_utils.load_proj_params(os.path.join(self.package_path,"configs",self.general_params["projector_calibration"]))
        self.resolution_codes    = {"HD2K"  :{"camera": sl.RESOLUTION.HD2K  , "projector" : {"width": int(2048), "height": int(1080)} },
                                    "HD1080":{"camera": sl.RESOLUTION.HD1080, "projector" : {"width": int(1920), "height": int(1080)} },
                                    "HD720" :{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(1280), "height": int(720)}  },
                                    "VGA"   :{"camera": sl.RESOLUTION.VGA   , "projector" : {"width": int(640),  "height": int(320)}  },
                                    "Custom":{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(1920), "height": int(1080)} }}
        
        ######################################################
        self.plane_3d         = intpro_utils.get_plane(width = 1.5,length = 0.8, origin_x = 0, origin_y = 0)
        # self.plane_3d         = intpro_utils.get_plane(width = 1.5,length = 1.5, origin_x = 0, origin_y = 0)
        self.points_2d        = intpro_utils.project3d_2d(K = self.projector_intrinsic, D = self.projector_params["distortion"],
                                                          tf_mat = self.home_tf, points = self.plane_3d) 
        self.proj_w           = self.general_params["projector_width"]
        self.proj_h           = self.general_params["projector_height"]
        self.projector_screen = np.ones((self.proj_h, self.proj_w, 3), dtype=np.uint8)
        self.goal_image       = np.ones(self.projector_screen.shape, dtype=np.int32)
        self.robot_goals      = []    
                     
        # given transformation of table plane wrt camera is going to be constant, we only have to calculate homography once
        self.H_home                = self.get_H(points_2d=self.points_2d, width=500, height=750) # for unity
        
        # self.H_home = self.get_H(points_2d=self.points_2d) # for gazebo
        self.static_flag      = False
        
        # red and green text for static
        self.red_tex   = self.get_red_tex()
        self.green_tex = self.get_green_tex()
        self.all_goals = self.get_all_goals()
        self.goal_plane3d   = intpro_utils.get_plane(width=0.1, length=0.1, origin_x=-0.1,origin_y= 0)       
        self.goal_points2d = {}
        for g_idx, g in enumerate(self.all_goals):
            p2d = intpro_utils.project3d_2d(K = self.projector_intrinsic,
                                            D = self.projector_params["distortion"],
                                            tf_mat= g,
                                            points = self.goal_plane3d)
            self.goal_points2d.update({(g_idx+1):[p2d,self.get_H(points_2d=p2d, height=self.red_tex.shape[0], width=self.red_tex.shape[1])]})
            
        ######################################################
        self.robot_goal_nums = kwargs["robot_goals_num"]
        self.pattern_image = None 
        self.downtop_img   = Image()
        self.unity_img     = Image()
        self.opegl_image   = Image()
        self.opegl_plane   = Image()
        self.rgb_planes    = Image()
        self._cvbridge = CvBridge()
        
        rospy.Subscriber("/downtop_camera/image_raw",Image,self.downtop_cam_callback)
        rospy.Subscriber("/shadow_image",Image,self.unity_cam_callback)
        rospy.Subscriber("/pov_image",Image,self.opengl_callback)
        rospy.Subscriber("/pov_plane",Image,self.opengl2_callback)
        rospy.Subscriber("/rgb_planes",Image,self.rgbplanes_callback)
        
        
        self.flag = False        
        self.user_disp_pub = rospy.Publisher("user_display",Image,queue_size=1)
        self.order = {"r":0, "b":1, "g":2, "c": 3}
        self.start_time = 0.0
        self.delta_time = 0.5
        self.c_goal     = -100

        ######################################################       
        self.user_dir     = os.path.join("/share/iros_1_dataset",kwargs["username"])
        if not os.path.isdir(self.user_dir):
            os.mkdir(self.user_dir)
            
        ######################################################      
        #  start opencv window
        self.projector_name = self.general_params["projector_name"]
        # cv2.namedWindow(self.projector_name,cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(self.projector_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        # cv2.moveWindow(self.projector_name,self.general_params["x_offset"],self.general_params["y_offset"])
        
        ######################################################       
        # init ros paramters
        rospy.set_param("start_user_exp",False)
        rospy.set_param("stop_user_exp",False)
        rospy.set_param("new_mode",True)
        rospy.set_param("stop_saving",True)
        rospy.set_param("current_mode","unkonwn")
        rospy.set_param("user_init",True)
        rospy.set_param("user_name",kwargs["username"])
        rospy.set_param("user_dir",self.user_dir)
        
        self.modes = {0:"static_mode", 1:"dynamic_mode", 2: "dual_mode", 3:"noproj_mode"}        
        self.current_mode     =  None
        self.current_mode_idx = 0
    
        ######################################################
        self.robot_goals_ = [np.array([12, 1,17,15,10,8]),
                             np.array([6,7,11,2,16,9]),
                             np.array([18,1,16,14,5,3]),
                             np.array([6,13,4,15,17,2])]
        
        ######################################################       
        # misc parameters
        self.counter_buff = 1e6
        self.tf_mat_1 = np.eye(4)
        self.tags = []
        self.goal_mat = np.eye(4)
        
        
        ###################################################### 
        # moving average params
        self.mvavg_buff = 3
        self.mvavg_counter = 0
        self.src_mat_mvavg = np.zeros((self.mvavg_buff,4,2))
        self.tmp_src_mat  = np.zeros((4,2))

        return
    
    def rospy_thread(self) -> None:
        rospy.spin()
      
    def downtop_cam_callback(self, msg:None) -> None:
        self.downtop_img = msg
    
    def unity_cam_callback(self, msg:None) -> None:
        self.unity_img = msg
    
    def opengl_callback(self,msg):
        self.opegl_image = msg

    def opengl2_callback(self,msg):
        self.opegl_plane = msg
        
    def rgbplanes_callback(self,msg):
        self.rgb_planes = msg
    
    def smartwatch_callback(self,msg):
        self.smartwatch_data = msg 
             
    def init_projector_screen(self) -> None:
        self.projector_screen = np.ones((self.general_params["projector_height"], self.general_params["projector_width"], 3), dtype=np.uint8)
        
       
    def get_red_tex(self):
        return cv2.imread(os.path.join(self.package_path,"textures","red_round.png"))
    
    def get_green_tex(self):
        return cv2.imread(os.path.join(self.package_path,"textures","green_round.png"))
    
    def moving_average(self,array, window_size = 2, get_mean_mvavg =True ):
        if len(array) <= window_size:
            print("need bigger array than window size or reduce window size")
            return array
        idx = 0
        array_mvavg = []
        while idx < (len(array)-window_size):
            window_avg_xy1 = np.mean(array[idx:idx+window_size,0,:],axis=0)
            window_avg_xy2 = np.mean(array[idx:idx+window_size,1,:],axis=0)
            window_avg_xy3 = np.mean(array[idx:idx+window_size,2,:],axis=0)
            window_avg_xy4 = np.mean(array[idx:idx+window_size,3,:],axis=0)
            array_mvavg.append([window_avg_xy1,window_avg_xy2,window_avg_xy3,window_avg_xy4])
            idx += 1
            
        if get_mean_mvavg:
            array_mvavg = np.array(array_mvavg)
            array_mvavg = np.mean(array_mvavg,axis=0)
            print(array_mvavg.shape)
            # array_mvavg = np.mean(array_mvavg)
            
        return array_mvavg
    
    def get_all_goals(self):
        # first six goals
        ref_goals = []
        for i in range(6):
            tmp = np.load(os.path.join(self.package_path,"files",f"{i+1}.npy"))
            if i == 5:
                tmp[0,-1] += 0.015
            ref_goals.append(tmp)
        # now relative goals to initial six goals
        all_goals = []
        for j in range(1,3):
            for rg in ref_goals:
                mat = compose([j*0.15,0,0],np.eye(3),[1,1,1])
                mat = np.matmul(rg,mat)
                all_goals.append(mat)
        
        all_goals = ref_goals + all_goals
        all_goals = np.array(all_goals).reshape(-1,4,4)
        return all_goals
        
    def get_H(self, points_2d, height = 1500, width = 1000):
     
        src_mat = np.array([[0     ,  0],
                            [width ,  0],
                            [width ,  height],
                            [0     ,  height]],
                            dtype = np.int32)
        H , _ = cv2.findHomography(srcPoints=src_mat, dstPoints= points_2d)
        return H
    
    def hflip_image(self,image):
        image = cv2.flip(image,0)
        return image
    
    def vflip_image(self,image):
        image = cv2.flip(image,1)
        return image
    
    def rclock_image(self,image):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image
    
    def rcounterclock_image(self,image):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    
    def warped_persp(self, image, H, width, height):
        warped = cv2.warpPerspective(image, H, (width,height)).astype(np.uint8) 
        return warped
    
    def add_images(self, image1, image2, alpha=1, beta=1, gamma=1):
        return cv2.addWeighted(image1, alpha, image2, beta, gamma)
    
    def get_warped_image(self, texture_image: np.ndarray, points_2d: np.ndarray, rotate: bool = False, hflip: bool = False, vflip: bool = False) -> np.ndarray:
        h,w,c = texture_image.shape
        if rotate:
            # M = cv2.getRotationMatrix2D((int(w/2),int(h/2)),rot_angle,1)
            # texture_image = cv2.warpAffine(texture_image,M,(w,h))
            texture_image = cv2.rotate(texture_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        if hflip:
            texture_image = cv2.flip(texture_image,0)
        
        if vflip:
            texture_image = cv2.flip(texture_image,1)
        src_mat = np.array([[0                      ,                0],
                            [texture_image.shape[1] ,                0],
                            [texture_image.shape[1] , texture_image.shape[0]],
                            [0                      , texture_image.shape[0]]],
                            dtype = np.int32)
        H , _ = cv2.findHomography(srcPoints=src_mat, dstPoints= points_2d)
        warped_image =  cv2.warpPerspective(texture_image,
                                            H,
                                            (self.general_params["projector_width"],
                                            self.general_params["projector_height" ])).astype(np.uint8)     
        return warped_image
    

    def get_contour(self, image, channel = 0,th1 = 127, th2 = 255):
        if len(image.shape) ==3: 
            img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype("uint8")
        else:
            img = image
        ret, thresh = cv2.threshold(img, th1, th2, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def get_centroid(self, img):
        cnt = self.get_contour(img)
        centroids = []
        for c in cnt:
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append([cx,cy])
        
        return centroids

    
    def is_match(self,cord_1, cord_2, thresh = 10):
        x_dif = abs(cord_1[0] -cord_2[0])
        y_dif = abs(cord_1[1] -cord_2[1])
        if x_dif <thresh and y_dif < thresh:
            return True
        else:
            return False
    
    def apply_img_conversion(self,x,y):
        curr_x,curr_y= 800,600
        rgb_x, rgb_y = 1920, 1080
        x = int(x * rgb_x /curr_x)
        y = int(y * rgb_y /curr_y)
        return x, y
        
    def compute_unity_plane(self, cv_rgb):
        # cv_rgb = cv2.resize(cv_rgb, (800,600))
        
        c1 = np.where(cv_rgb[:,:,0] == 255,cv_rgb[:,:,0],0)
        c2 = np.where(cv_rgb[:,:,1] == 255,cv_rgb[:,:,1],0)
        c3 = np.where((c1 == c2), c2,0) # cyan
        c4 = np.where((c1 != c3), c1,0) # blue
        c5 = np.where((c2 != c3), c2,0) # green
        c6 = cv_rgb[:,:,2] # red

        red_cent   = self.get_centroid(c6)
        green_cent = self.get_centroid(c5)
        blue_cent  = self.get_centroid(c4) 
        cyan_cent  = self.get_centroid(c3) 
        cents = {'r':red_cent, 'g':green_cent, 'b':blue_cent, 'c':cyan_cent}
        matched_cents_x = [0,0,0,0]
        matched_cents_y = [0,0,0,0]
            
        for k,v in self.order.items():
            # x , y = self.apply_img_conversion(cents.get(k)[0][0],cents.get(k)[0][1] )
            x, y = cents.get(k)[0][0], cents.get(k)[0][1]
            matched_cents_x[v] = x
            matched_cents_y[v] = y

        x_ = np.array(matched_cents_x, dtype= int).reshape(-1,1)
        y_ = np.array(matched_cents_y , dtype= int).reshape(-1,1)
        new_src_mat = np.hstack((x_,y_))
        src_mat = np.array([new_src_mat[0,:],new_src_mat[3,:],new_src_mat[1,:],new_src_mat[2,:]])
        return src_mat
 
    def show_all(self):
        imgs = np.copy(self.projector_screen)
        for k, goal_data in self.goal_points2d.items():
            img = self.warped_persp(self.red_tex, goal_data[1],self.proj_w,self.proj_h)
            imgs = self.add_images(imgs,img)
        return imgs
    
    def dynamic_mode(self):
        # robot_shadow = self._cvbridge.imgmsg_to_cv2(self.downtop_img,desired_encoding="bgr8")
        if len(self.unity_img.data) == 0:
            print("waiting for shadow image msg from unity...")
            return self.projector_screen
        robot_shadow = self._cvbridge.imgmsg_to_cv2(self.unity_img,desired_encoding="bgr8")
        robot_shadow = self.rcounterclock_image(robot_shadow)
        robot_shadow = self.hflip_image(robot_shadow)
        warped_image = self.warped_persp(robot_shadow,self.H_home,self.proj_w,self.proj_h)
        return warped_image
    
    def static_mode(self):
        goal = rospy.get_param("goal_id",default=100)
        if goal == self.c_goal and self.start_recording: 
            self.start_time = time.time()
            self.start_recording = False
         
        if self.c_goal is not goal:
            self.c_goal = goal
            self.start_recording = True
            
        delta = (time.time() - self.start_time) 
        # if  delta >= (2.5 * rospy.get_param("time_delta",default=1.0)):
        # if  not (rospy.get_param("start_static",default=False) and delta > 0.5):
        if  not (rospy.get_param("start_static",default=False)):
            return self.projector_screen
        
        print(f"delta: {delta}")
        
        if goal is None:
            return self.projector_screen
        
        g_data = self.goal_points2d.get(goal)
        if g_data is None:
            return self.projector_screen
        
        warped_image = self.warped_persp(self.red_tex, g_data[1],self.proj_w,self.proj_h)
        
        # if self.static_flag:
        #     warped_image = self.warped_persp(self.red_tex, g_data[1],self.proj_w,self.proj_h)
        #     self.static_flag = False
            
        # if not self.static_flag:
        #     self.static_flag = True
        #     warped_image = self.warped_persp(self.green_tex, g_data[1],self.proj_w,self.proj_h)
            
        
        return warped_image
                
    def dual_mode(self):
        img1 = self.static_mode()
        img2 = self.dynamic_mode()
        return self.add_images(img1,img2)
                            
    def noproj_mode(self):
        self.init_projector_screen()
        return self.projector_screen
        
    def smartwatch_teleop(self):
        
        pass
    
    def exp_mv_avg(self, src_mat):
        
        out_mat = 0.8 * src_mat + 0.2 * self.tmp_src_mat
        self.tmp_src_mat = src_mat
        return out_mat
        
        
    def projection_3d(self):
        opengl_image = CvBridge().imgmsg_to_cv2(self.opegl_image,desired_encoding="bgr8")
        rgb_planes   = CvBridge().imgmsg_to_cv2(self.rgb_planes,desired_encoding="bgr8")   
        src_mat = self.compute_unity_plane(rgb_planes)
        src_mat = self.exp_mv_avg(src_mat)
        
        # if self.mvavg_counter == self.mvavg_buff-1:
        #     self.src_mat_mvavg = np.roll(self.src_mat_mvavg,-1,axis=0)
        # else:
        #     self.mvavg_counter += 1 
        # self.src_mat_mvavg[self.mvavg_counter] = src_mat
        # src_mat_mvavg = self.moving_average(self.src_mat_mvavg)
        # if self.mvavg_counter != self.mvavg_buff-1:
        #     return self.projector_screen
        # H , _ = cv2.findHomography(srcPoints=src_mat_mvavg, dstPoints= self.points_2d)
        H , _ = cv2.findHomography(srcPoints=src_mat, dstPoints= self.points_2d)
        warped_image =  cv2.warpPerspective(opengl_image,
                                            H,
                                            (self.general_params["projector_width"],
                                            self.general_params["projector_height" ])).astype(np.uint8)   
        
        # newcammatrix,_ = cv2.getOptimalNewCameraMatrix(self.projector_intrinsic,self.projector_params["distortion"],(warped_image.shape[1],warped_image.shape[0]),
        #                                              1, (warped_image.shape[1],warped_image.shape[0]))
        # print(type(newcammatrix))
        # undistorted_warped = cv2.undistort(warped_image,
        #                                    self.projector_intrinsic,
        #                                    np.array(self.projector_params["distortion"]),
        #                                    None,
        #                                    np.array(newcammatrix))
        cv2.imshow("img",cv2.resize(warped_image,(800,600)))
        
        return warped_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--username",type=str,default="")
    parser.add_argument("--save_goals","-sg",type=int, default=0)
    parser.add_argument("--robot_goals_num","-rgn",type=int, default=6)
    parser.add_argument("--display","-d",type=int, default=0)
    
    args = parser.parse_args()
    kwargs = vars(args)
    intpro = INTPRO(**kwargs)  
    rate = rospy.Rate(30)
    print("waiting 1 sec")
    rospy.sleep(1)
    
    modes_funcs = [intpro.static_mode, intpro.dynamic_mode, intpro.dual_mode, intpro.noproj_mode]
    modes = ["static_mode", "dynamic_mode", "dual_mode", "noproj_mode"]
    mode_id = -1
    display = bool(args.display)
    while not rospy.is_shutdown():
        try:
            t1 = time.time()
            if rospy.get_param("new_mode",default=False):
                mode_id = np.random.choice([0,1,2,3])
                rospy.set_param("mode_id",modes[mode_id])
                rospy.set_param("user_init",True)
                rospy.set_param("current_mode",intpro.modes.get(mode_id))
                rospy.set_param("new_mode",False)
            
            if display:
                cv2.namedWindow(intpro.projector_name,cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(intpro.projector_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.moveWindow(intpro.projector_name,intpro.general_params["x_offset"],intpro.general_params["y_offset"])
                cv2.imshow(intpro.projector_name,intpro.projection_3d())
            
            # uncomment below to run iros exepriment
            ########################################################
            # print(f"running {intpro.modes.get(mode_id)}")
            # cv2.imshow(intpro.projector_name,modes_funcs[mode_id]())
            # if rospy.get_param("show_all_goals",default=False):
            #     cv2.imshow(intpro.projector_name,intpro.show_all())
            ########################################################
            cv2.waitKey(1)
            rate.sleep()
            
            print(f"rate: {1 /np.round((time.time()-t1),2)} hz")
        except BaseException as e:
            print(traceback.format_exc())
            print(f"Got error in main loop: {e}")
        