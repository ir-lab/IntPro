
from decimal import DecimalTuple
from dis import dis, disco
from distutils.ccompiler import gen_lib_options
import os
from queue import Empty 
import sys
import time
import cv2
import numpy as np
import pyzed.sl as sl
from transforms3d.quaternions import quat2mat
from transforms3d.euler import quat2euler, euler2mat, mat2euler
from transforms3d.affines import compose
from intpro_utils import intpro_utils
from pupil_apriltags import Detector
from threading import Thread, Lock
from time import perf_counter
import datetime
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch
import os
import matplotlib.pyplot as plt
import time
import random
from skimage import io, transform
from models.CorrectionModel import CorrectionModel
from utils.DataSet import DataSet

class projection_mapping:
    """
    Doc: 
        Note: Supports ZED 2 Camera (SDK 3.7) and Optomo Projector
        
        1) Given calibration file check and improve calibration file
    """
    def __init__(self) -> None:
        
        self.this_file_path = os.path.abspath(__file__)
        self.ip_dir = os.path.dirname(os.path.abspath(os.path.join(self.this_file_path,"../.")))

        # misc parameters
        self.counter_buff = 1e6
        self.tf_mat_1 = np.eye(4)
        
        # general params 
        self.general_params = intpro_utils.load_yaml(os.path.join(self.ip_dir,"configs/general_params.yaml"))
        self.resolution_codes = {"HD2K"  :{"camera": sl.RESOLUTION.HD2K  , "projector" : {"width": int(2048), "height": int(1080)} },
                                 "HD1080":{"camera": sl.RESOLUTION.HD1080, "projector" : {"width": int(1920), "height": int(1080)} },
                                 "HD720" :{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(1280), "height": int(720)}  },
                                 "VGA"   :{"camera": sl.RESOLUTION.VGA   , "projector" : {"width": int(640),  "height": int(320)}  },
                                #  "Custom":{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(4096), "height":  int(2160)}  }}
                                 "Custom":{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(1920), "height":  int(1080)}  }}
                                #  "Custom":{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(800), "height":  int(600)}  }}
        # initilize projector
        self.projector_params = intpro_utils.load_proj_params(os.path.join(self.ip_dir,"configs", self.general_params["projector_calibration"]))
        # self.projector_params = intpro_utils.load_proj_params(os.path.join(self.ip_dir,"configs/calibration_cropped.yml"))
        # self.projector_params = intpro_utils.load_proj_params(os.path.join(self.ip_dir,"configs/calibration_hd.yml"))
        # self.projector_params = intpro_utils.load_proj_params(os.path.join(self.ip_dir,"configs/calibration.yml"))
        if self.general_params["use_saved_params"]:
            self.projector_params["intrinsic"] = np.load(os.path.join(self.ip_dir,"files/projector_intrinsic.npy"))
            self.projector_params["extrinsic"] = np.load(os.path.join(self.ip_dir,"files/projector_extrinsic.npy"))

        self.projector_screen = np.zeros(shape=(self.resolution_codes[self.general_params["projector_res"]]["projector"]["height"],
                                                self.resolution_codes[self.general_params["projector_res"]]["projector"]["width"],3),
                                         dtype= np.uint)
        
        self.show_projector_screen = True
        
        # initialize zed camera
        self.frame_    = self.general_params["zed_frame"]
        self.zed       = sl.Camera()
        
        
        self.zed_image = sl.Mat()
        self.zed_init_params = sl.InitParameters()
        self.zed_init_params.camera_resolution = self.resolution_codes[self.general_params["zed_res"]]["camera"]
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

        # initialize tag detector
        self.keep_detection = True
        self.tag_detector = Detector(families=self.general_params["tag_families"],
                                     nthreads=self.general_params["nthreads"],
                                     quad_decimate=self.general_params["quad_decimate"],
                                     quad_sigma=self.general_params["quad_sigma"],
                                     refine_edges=self.general_params["refine_edges"],
                                     decode_sharpening=self.general_params["decode_sharpening"],
                                     debug=self.general_params["debug"])
        
        self.cv_key = cv2.waitKey(1)
        self.tags = None
        self.tag_thread = Thread(target=self.detect_tag,args=())
        self.tag_thread.start()
        
        print("Initialized framework.......\nsleeping for 1 second.")
        time.sleep(1)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = self.init_model("/share/auto_correct_intproj/2999.pth")
        print("loaded model")
        self.start_0 = True
        self.start_1 = True
        self.counter_ = 0
        self.got_tf_mat = False
        self.tf_mat_5 = np.eye(4)
        self.tmp_tf = np.zeros((4,4))
        self.data_dir =os.path.join("/share/predicted_dataset",self.get_timestamp())
        os.mkdir(self.data_dir)
        
    def reinit_projector_screen(self):
        self.projector_screen = np.zeros(shape=(self.resolution_codes[self.general_params["projector_res"]]["projector"]["height"],
                                                self.resolution_codes[self.general_params["projector_res"]]["projector"]["width"],3),
                                         dtype= np.uint)
        
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
    
        
    def get_rectpoints(self, corners:list) -> list:
        x_max,y_max = np.max(corners,axis=0)
        x_min,y_min = np.min(corners,axis=0)
        return [(int(x_min),int(y_min)),(int(x_max),int(y_max))]
    
    def crop_image(self,image):
        
        sh_image = image[-720:, int(image.shape[1]/2)-200:int(image.shape[1]/2)+440 ,:]
        return sh_image

    @intpro_utils.measure_performance
    def detect_tag(self) -> None:
        zed_params    = self.zed_params["params"]
        freq_buff = list()

        while self.keep_detection:
            try:
                
                if self.zed.grab(sl.RuntimeParameters()):
                    start_time = perf_counter()      
                    self.zed.retrieve_image(self.zed_image,sl.VIEW.LEFT if self.frame_=="left" else sl.VIEW.RIGHT)
                    image = self.zed_image.get_data()
                    image = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
                    
                    # zed_params.cx -= 200
                    # zed_params.cy -= image.shape[0]-720
                    # image = self.crop_image(image)
                    
                    # cv2.imshow("tf",image)
                    # cv2.waitKey(1)
                    self.tags = self.tag_detector.detect(image,self.general_params["estimate_pose"],
                                                    [zed_params.fx,zed_params.fy,zed_params.cx,zed_params.cy],                                                    
                                                    self.general_params["tag_size"])
                    stop_time = perf_counter()
                    
                    freq_buff.append((1/(stop_time - start_time)))                    
                    if len(freq_buff) >= self.counter_buff:
                        tmp = freq_buff[-1]
                        freq_buff = []
                        freq_buff.append(tmp)
                    # print(f"Current avg freq [{np.round(np.mean(freq_buff),2)}] hz")          
            except Exception as e:
                print(f"Got Error: {e}")
    
    def filter_calibration(self):
        self.reinit_projector_screen()
        image = np.copy(self.zed_image.get_data())
        image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
        # time.sleep(1)
        x_offset_origin = np.arange(start=1,stop=5,step=2)
        y_offset_origin = np.arange(start=0,stop=2,step=2)
        # x_offset_origin = np.random.choice(np.arange(start=1,stop=10,step=2))
        # y_offset_origin = np.random.choice(np.arange(start=0,stop=6,step=1))
        # x_offset_origin = 1
        # y_offset_origin = 0
        change_color = 0
        tmp_proj_screen = []
        tmp_cam_screen  = []
        
        for tag in self.tags:
            if tag.tag_id == self.general_params["test_id"]:
                pose_R = tag.pose_R
                pose_t = tag.pose_t.reshape(-1)
                tf_mat = intpro_utils.get_compose(trans=pose_t,
                                                rot_mat=pose_R)
                for i in x_offset_origin:
                    for j in y_offset_origin:
                        plane_3d = intpro_utils.get_plane(width=0.12192,length=0.12192, 
                                                        origin_x= -0.15-(0.12192/2 * i), 
                                                        origin_y= 0 * -j)
                        # plane_3d = intpro_utils.get_plane(width=0.12192,length=0.12192, 
                        #                                   origin_x= -0.071-(0.12192/2 * x_offset_origin), 
                        #                                   origin_y= 0.12192 * -y_offset_origin)        
                        # plane_3d = intpro_utils.get_plane(width=0.12192,length=0.12192, origin_x= 0.0, origin_y=0.0)
                                # if self.tags:
                
                        
                        ######### Manipulate the projector params [fixed] #########                      
                        self.cv_key = cv2.waitKey(1)
                        
                        if self.general_params["use_keyevent"]:
                            intpro_utils.keyEvent(cv_key= self.cv_key,
                                                tf_mat = self.tf_mat_1,
                                                tmp_projector_intrinsic  = self.projector_params["intrinsic" ],
                                                tmp_projector_distortion = self.projector_params["distortion"],
                                                ext_offset=0.5)

                        if not self.general_params["use_saved_params"]:
                            tf_mat_2 = np.matmul(self.projector_params["extrinsic"],tf_mat)
                            tf_mat_2 = np.matmul(tf_mat_2, self.tf_mat_1)
                            projector_ext_new = np.matmul(tf_mat_2,np.linalg.inv(tf_mat))
                            
                        else:
                            tf_mat_2 = np.matmul(self.projector_params["extrinsic"],tf_mat)
                            tf_mat_2 = np.matmul(tf_mat_2, self.tf_mat_1)
                            projector_ext_new = np.matmul(tf_mat_2,np.linalg.inv(tf_mat))
                            
                        test_mat = compose([-0.23, 0.073, 0.05],euler2mat(0,0.0175,0),[1,1,1])
                        test_mat = np.matmul(test_mat,tf_mat)
                        ################################
                        print(tf_mat_2[0:3,-1])
                        print(tf_mat[0:3,-1])
                        # print(self.projector_params["extrinsic"][0:3,-1])
                        if self.general_params["save_proj_ext"]:
                            np.save(os.path.join(self.ip_dir,f"files/projector_extrinsic.npy"),projector_ext_new)
                        else:
                            print("Not saving projector extrinisic")
                            
                        if self.general_params["save_proj_int"]:
                            np.save(os.path.join(self.ip_dir,f"files/projector_intrinsic.npy"),self.projector_params["intrinsic"])
                        else:
                            print("Not saving projector intrinsic")
                        
                        points_2d_cam = intpro_utils.project3d_2d(K = self.zed_params["intrinsic"], 
                                                                D = self.zed_params["distortion"],
                                                                tf_mat = tf_mat,
                                                                points =  plane_3d)      
                        
                        
                        # points_2d = intpro_utils.project3d_2d(K = self.projector_params["intrinsic"], 
                        #                                     D = self.projector_params["distortion"],
                        #                                     tf_mat = test_mat,
                        #                                     points = plane_3d) 
                        points_2d = intpro_utils.project3d_2d(K = self.projector_params["intrinsic"], 
                                                            D = self.projector_params["distortion"],
                                                            tf_mat = tf_mat_2,
                                                            points = plane_3d)     
                        
                        
                        texture_mat_proj = np.ones((500,500,3), dtype=np.uint8)
                        texture_mat_cam  = np.ones((500,500,3), dtype=np.uint8)
                        if change_color%2 == 0:
                            texture_mat_cam[:,:,1]  *= 255
                            texture_mat_proj[:,:,1] *= 255
                        
                        else:
                            texture_mat_cam[:,:,2]  *= 255
                            texture_mat_proj[:,:,2] *= 255
                            
                        change_color += (1+(1+j)*i)
                        
                        src_mat = np.array([[0  , 0],
                                            [500, 0],
                                            [500, 500],
                                            [0  , 500]], dtype= np.float32)
                                            
                        H     , _ = cv2.findHomography(srcPoints=src_mat, dstPoints=points_2d)
                        H_cam , _ = cv2.findHomography(srcPoints=src_mat, dstPoints=points_2d_cam)
                        
                        projector_image   = cv2.warpPerspective(texture_mat_proj,
                                                                H,
                                                                (self.projector_screen.shape[1],
                                                                self.projector_screen.shape[0]))
                        
                        cam_image   = cv2.warpPerspective(texture_mat_cam,
                                                        H_cam,
                                                        (image.shape[1],
                                                        image.shape[0]))
                        
                        tmp_proj_screen.append(projector_image)
                        tmp_cam_screen.append(cam_image)
                        
            
        for idx in range(len(tmp_cam_screen)):
            self.projector_screen = cv2.addWeighted(self.projector_screen,
                                                    1.0,
                                                    tmp_proj_screen[idx],
                                                    # projector_image
                                                    1.0,
                                                    gamma=0.0,
                                                    dtype=cv2.CV_8U)
            image = cv2.addWeighted(image,
                                    1.0,
                                    tmp_cam_screen[idx],
                                    # cam_image,
                                    1.0,
                                    gamma=0.0,
                                    dtype=cv2.CV_8U)
        
        
        self.projector_screen = self.projector_screen.astype(np.uint8)
        # self.projector_screen = cv2.flip(self.projector_screen,0)
        if self.cv_key == ord(' '):
            self.show_projector_screen = True if self.show_projector_screen else False
        if self.show_projector_screen:
            intpro_utils.show_projector_screen(x_offset = self.general_params["x_offset"],
                                               y_offset = self.general_params["y_offset"],
                                               image = self.projector_screen)
        cv2.imshow("rendered_image",self.projector_screen)
        cv2.imshow("cam_image",image)
    
    def get_rectpoints(self, corners):
        x_max,y_max = np.max(corners,axis=0)
        x_min,y_min = np.min(corners,axis=0)

        return (int(x_min),int(y_min)), (int(x_max),int(y_max))
    
    def get_frames(self,tf_mat,delta_x =0.01, delta_y=0.01):
        tf_frames = []
        goal_xyz = tf_mat[0:3,-1]
        goal_rpy = mat2euler(tf_mat[0:3,0:3])
        # data = np.random.randint(1,30)
        data = 5
        
        choice =  np.random.rand()
        if choice < 0.25:
            x_offset = goal_xyz [0] + data*delta_x
            y_offset = goal_xyz [1] + data*delta_y
            for i in range(data):
                tf_frame = compose([x_offset,y_offset,goal_xyz[-1]],tf_mat[0:3,0:3],[1,1,1])
                tf_frames.append(tf_frame)
                x_offset -= delta_x
                y_offset -= delta_y
        elif choice >=0.25 and choice < 0.50:
            x_offset = goal_xyz [0] - data*delta_x
            y_offset = goal_xyz [1] - data*delta_y
            for i in range(data):
                tf_frame = compose([x_offset,y_offset,goal_xyz[-1]],tf_mat[0:3,0:3],[1,1,1])
                tf_frames.append(tf_frame)
                x_offset += delta_x
                y_offset += delta_y
        elif choice >=0.5 and choice < 0.75:
            x_offset = goal_xyz [0] + data*delta_x
            y_offset = goal_xyz [1] - data*delta_y
            for i in range(data):
                tf_frame = compose([x_offset,y_offset,goal_xyz[-1]],tf_mat[0:3,0:3],[1,1,1])
                tf_frames.append(tf_frame)
                x_offset -= delta_x
                y_offset += delta_y
        else:
            x_offset = goal_xyz [0] - data*delta_x
            y_offset = goal_xyz [1] + data*delta_y
            for i in range(data):
                tf_frame = compose([x_offset,y_offset,goal_xyz[-1]],tf_mat[0:3,0:3],[1,1,1])
                tf_frames.append(tf_frame)
                x_offset += delta_x
                y_offset -= delta_y
                

        tf_frames.append(tf_mat)
        return tf_frames
        
    def get_timestamp(self):
        time_stamp = str(datetime.datetime.now()).replace(" ","_")
        time_stamp = time_stamp.replace("-","_")
        time_stamp = time_stamp.replace(":","_")
        time_stamp = time_stamp.replace(".","_")
        return time_stamp
    
    
    def get_predicted_tf(self,img):
        img = self.preprocess_img(img)
        delta_pos_pred = self.model(img)
        delta_pos_pred = self.postprocess_pred(delta_pos_pred) / 7.0
        return delta_pos_pred


    def preprocess_img(self,img):
        return torch.tensor(img[364:364+700,732:732+700,:3] / 255, dtype=torch.float32).unsqueeze(0).to(self.device)

    def postprocess_pred(self,pred):
        return pred.detach().cpu().numpy()[0].reshape((4, 4))

    
    def init_model(self,ckpt):

        # load model
        model = CorrectionModel()
        model.load_state_dict(torch.load(ckpt), strict=False)
        model = model.to(self.device)

        return model
    def project_pattern(self):
        self.reinit_projector_screen()
        image =  np.copy(self.zed_image.get_data())
        image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
        if self.tags and not self.got_tf_mat:
            for tag in self.tags:
                if tag.tag_id == self.general_params["test_id"]:
                    break
            pose_R = tag.pose_R
            pose_t = tag.pose_t.reshape(-1)
            corners  = tag.corners
            tf_mat = intpro_utils.get_compose(trans=pose_t,
                                        rot_mat=pose_R)
            self.tf_mat_5 = tf_mat
            self.got_tf_mat = True
            tf_frames = self.get_frames(tf_mat=self.tf_mat_5)
            
        elif not self.got_tf_mat:
            # if self.start_0:
            return  False
        
        
        print(" project pattern")
        # time.sleep(2)
            
        if self.general_params["use_keyevent"]:
            intpro_utils.keyEvent(cv_key= self.cv_key,
                                tf_mat = self.tf_mat_1,
                                tmp_projector_intrinsic  = self.projector_params["intrinsic" ],
                                tmp_projector_distortion = self.projector_params["distortion"],
                                ext_offset=1, rotoffSet= 0.1)
        
        if self.general_params["load_home_tf"]:
            tf_mat_2 = np.load(os.path.join(self.ip_dir,"files","home_tf.npy"))
        else:
            tf_mat_2 = np.matmul(self.projector_params["extrinsic"],tf_mat)
            tf_mat_2 = np.matmul(tf_mat_2, self.tf_mat_1)
            
        texture_image = np.ones((500,500,3), dtype=np.uint8)
        texture_image[:,:,2] *= 255
        
        cam_texture_image = np.ones((500,500,3), dtype=np.uint8)
        cam_texture_image[:,:,:] *= 255
        # texture_image = cv2.circle(texture_image,(250,250),250,(0,0,250),-1)
        # cv2.imshow("texture image",texture_image)
        # texture_image = np.load(self.ip_dir,"textures/pattern_1.png")
        # plane_3d = intpro_utils.get_plane(width=0.12192,length=0.12192, 
                                        #   origin_x= 0.071+(0.12192/2 ), 
                                        #   origin_y= 0)
                                        
                                        
        
                
        if self.general_params["save_home_tf"]:
            np.save(os.path.join(self.ip_dir,"files","home_tf"), tf_mat_2)
        
        plane_3d = intpro_utils.get_plane(width=0.16210,length=0.16210, 
                                          origin_x= 0, 
                                          origin_y= 0)
        new_intrinsic, roi  = cv2.getOptimalNewCameraMatrix(self.projector_params["intrinsic"], 
                                                       self.projector_params["distortion"], 
                                                       (self.projector_screen.shape[1],self.projector_screen.shape[0]),
                                                       1,
                                                       (self.projector_screen.shape[1],self.projector_screen.shape[0]))
        
        # points_2d = intpro_utils.project3d_2d(K = self.projector_params["intrinsic"], 
        #                                       D = self.projector_params["distortion"],
        #                                       tf_mat = tf_mat_2,
        #                                       points = plane_3d) 
        
        # tf_frames = self.get_frames(tf_mat=tf_mat)
        time_stamp =self.get_timestamp()
        
        # time.sleep(1)
        # for idx,tm_tf in enumerate(tf_frames):
            
        # time.sleep(1)
        
        
        points_2d_cam = intpro_utils.project3d_2d(K = self.zed_params["intrinsic"], 
                                                                D = self.zed_params["distortion"],
                                                                tf_mat = self.tf_mat_5,
                                                                # tf_mat = tf_mat,
                                                                points =  plane_3d)      
        
        time.sleep(0.1)
        image =  np.copy(self.zed_image.get_data())
        image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
        cv2.imshow("image",image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # predicted_tf[0:2,-1] /= 5.0
        tf_frames = np.copy(self.tf_mat_5)
        # if self.start_1:
        #     tf_frames[0,-1] -= 0.05
        #     tf_frames[1,-1] -= 0.05
        #     self.start_1 = False
            
        tf_frames[0,-1] = 0.15
        # tf_frames[1,-1] =
        if self.start_0:
            curr_tf = np.matmul(self.projector_params["extrinsic"],tf_frames)
        else:
            curr_tf = np.matmul(self.projector_params["extrinsic"],self.tf_mat_5)
            
        
        self.tmp_tf += self.get_predicted_tf(image)
        predicted_tf = self.tmp_tf + curr_tf
        
        print(predicted_tf)
        points_2d = intpro_utils.project3d_2d(K = self.projector_params["intrinsic"], 
                                            D = self.projector_params["distortion"],
                                            tf_mat = curr_tf if self.start_0 else predicted_tf,
                                            # tf_mat = np.matmul(self.projector_params["extrinsic"],predicted_tf),
                                            points = plane_3d) 
        src_mat = np.array([[0  , 0],
                            [500, 0],
                            [500, 500],
                            [0  , 500]], dtype= np.float32)
        H     , _ = cv2.findHomography(srcPoints=src_mat, dstPoints=points_2d)
        H_cam , _ = cv2.findHomography(srcPoints=src_mat, dstPoints=points_2d_cam)
        self.projector_screen   = cv2.warpPerspective(texture_image,
                                                H,
                                                (self.projector_screen.shape[1],
                                                self.projector_screen.shape[0]))
        
        
        new_intrinsic, roi  = cv2.getOptimalNewCameraMatrix(self.projector_params["intrinsic"], 
                                                    self.projector_params["distortion"], 
                                                    (self.projector_screen.shape[1],self.projector_screen.shape[0]),
                                                    1,
                                                    (self.projector_screen.shape[1],self.projector_screen.shape[0]))
        # print(new_intrinsic)
        # self.projector_screen, _ = cv2.undistort(self.projector_screen,
        #                                          self.projector_params["intrinsic"],
        #                                          self.projector_params["distortion"], 
        #                                          None,
        #                                          new_intrinsic)
        intrinsic = self.projector_params["intrinsic"]
        discoeff  = self.projector_params["distortion"]
        img  = cv2.undistort(self.projector_screen,intrinsic,
                                            discoeff,None,  
                                            newCameraMatrix=new_intrinsic)
        
        # self.projector_screen = img
        intpro_utils.show_projector_screen(x_offset = self.general_params["x_offset"],
                                        y_offset = self.general_params["y_offset"],
                                        image = self.projector_screen)
                                        #    image = cv2.resize(self.projector_screen,(4096,2160)))
        # print(f"working on frame: {idx}")
        # time.sleep(0.1)
        # image =  np.copy(self.zed_image.get_data())
        # image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
        # cv2.imshow("rendered_screen",self.projector_screen)
        cam_image   = cv2.warpPerspective(cam_texture_image,
                                        H_cam,
                                        (image.shape[1],
                                        image.shape[0]))
        # print(image.shape,cam_image.shape)
        # image = cv2.addWeighted(image,
        #                         1.0,
        #                         cam_image,
        #                         1.0,
        #                         gamma=0.0,
        #                         dtype=cv2.CV_8U)
        # p1, p2 = self.get_rectpoints(corners=corners)
        # image = cv2.rectangle(image,p1,p2,(0,0,255),3)
        gray_cam_image = cv2.cvtColor(cam_image,cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray_cam_image, 110, 255, cv2.THRESH_BINARY)
        contours, _= cv2.findContours(threshold, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            # print(approx)
            new_image = np.copy(image)
            new_image = cv2.fillConvexPoly(new_image,approx,(0,0,255))
            # mat = np.matmul(self.projector_params["extrinsic"],predicted_tf)
            
            cv2.imwrite(os.path.join(self.data_dir,f"{self.counter_}_raw.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.data_dir,f"{self.counter_}_plane.png"),new_image)
            np.save(os.path.join(self.data_dir,f"{self.counter_}_tf_mat.npy"),predicted_tf)
            
            
            # cv2.imwrite()
            # cv2.imshow("cam",new_image)
            # cv2.imshow("org_image",image)
            # print(tf_mat[0:3,-1])
            
        # cv2.imshow("cam",gray_cam_image)
        # time.sleep(1) if self.start_0 else ""
        self.counter_ += 1
        if self.counter_ >= 5:
            print("activating prediction....")
            self.start_0 = False
        self.cv_key = cv2.waitKey(100)
        return True
    
    def __del__(self):
        self.zed.close()
        
if __name__ == "__main__":
    pm = projection_mapping()
    
    demo = 0
    total_demo = 15
    while pm.keep_detection:
        # pm.filter_calibration()
        
        try:
            if pm.project_pattern():
            # print("detections")
                print(f"Finished {demo+1:01} / {total_demo:01} ")
                demo += 1
                # if demo == total_demo:
                #     print("\n Done!!!!!!")
                #     break
            # pm.filter_calibration()
            # image = pm.zed_image.get_data()
            # cv2.imshow("image",image)
            # cv2.waitKey(1)
            # if image.shape[0] > 0 and image.shape[1] > 0:
            #     if pm.tags:
            #         for tag in pm.tags:
            #             rect = pm.get_rectpoints(tag.corners)
            #             cv2.rectangle(image,rect[0],rect[1], (0,0,255), 1,)
            #     cv2.imshow("image",image)
            #     cv2.waitKey(1)
            # else:
            #     print("Did not grab a image frame!")
        except Exception as e:
            print(f"Error in main loop: {e}")
            # if KeyboardInterrupt:
            #     pm.tag_thread.join()
