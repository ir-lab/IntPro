#! /usr/bin/env python3
import sys
import os
import pyzed.sl as sl
import numpy as np
from typing import *

class ZED_CAMERA:
    
    def __init__(self, serial_num, frame_ = "left", resolution = "HD720", fps = 30, brightness = -1, contrast = -1, hue  = -1, saturation = -1, sharpness = -1, gain = -1, exposure = -1, whitebalance = -1) -> None:
        '''
        Initializes the ZED_CAMERA object with custom settings.

        Args:
            - serial_num (int)            : Serial number of the ZED camera.
            - frame_ (str, optional)      : A string indicating the camera frame to use (left or right). Defaults to "left".
            - resolution (str, optional)  : Resolution of the camera. Defaults to "HD720".
            - fps (int, optional)         : Frames per second. Defaults to 30.
            - brightness (int, optional)  : Brightness setting for the camera. Defaults to -1.
            - contrast (int, optional)    : Contrast setting for the camera. Defaults to -1.
            - hue (int, optional)         : Hue setting for the camera. Defaults to -1.
            - saturation (int, optional)  : Saturation setting for the camera. Defaults to -1.
            - sharpness (int, optional)   : Sharpness setting for the camera. Defaults to -1.
            - gain (int, optional)        : Gain setting for the camera. Defaults to -1.
            - exposure (int, optional)    : Exposure setting for the camera. Defaults to -1.
            - whitebalance (int, optional): White balance setting for the camera. Defaults to -1.

        Returns:
            None
        '''
        self.resolution_codes    = {"HD2K"  :{"camera": sl.RESOLUTION.HD2K  , "projector" : {"width": int(2048), "height": int(1080)} },
                                    "HD1080":{"camera": sl.RESOLUTION.HD1080, "projector" : {"width": int(1920), "height": int(1080)} },
                                    "HD720" :{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(1280), "height": int(720)}  },
                                    "VGA"   :{"camera": sl.RESOLUTION.VGA   , "projector" : {"width": int(640),  "height": int(320)}  },
                                    "Custom":{"camera": sl.RESOLUTION.HD720 , "projector" : {"width": int(1920), "height":  int(1080)}}}
        
        # initialize zed camera
        self.frame_           = frame_
        self.zed              = sl.Camera()        
        self.zed_init_params  = sl.InitParameters()

        if serial_num is not None:
            try:
                self.zed_init_params.set_from_serial_number(serial_num)
            except Exception as e:
                print(f"Unable to find ZED camera with serial number {serial_num} trying default ZED camear....")

        self.zed_init_params.camera_resolution = self.resolution_codes[resolution]["camera"]
        self.zed_init_params.camera_fps = fps
        self.zed_init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
        self.zed_init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
        self.zed_init_params.coordinate_units = sl.UNIT.METER  
        
        if not self.zed.open(self.zed_init_params):
            print("Unable to open ZED camera")
            exit(1)
            
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS,               brightness)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST,                 contrast)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE,                      hue)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION,               saturation)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS,                sharpness)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN,                     gain)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE,                 exposure)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, whitebalance)
        self.zed_params = self.get_zed_params()
            
        return
    
    def get_zed_params(self) -> dict:
        '''
        Retrieves the camera parameters for either the left or right camera.

        Returns:
            dict: A dictionary containing the intrinsic and distortion parameters of the camera.
        '''
        
        params = self.zed.get_camera_information().camera_configuration.calibration_parameters
        if self.frame_ == "left":
            params = params.left_cam
        else:
            params = params.right_cam        
        intrinsic  = np.array([[params.fx, 0.0       , params.cx],
                               [0.0      , params.fy , params.cy],
                               [0.0      , 0.0       , 1.0      ]])
        
        distortion = np.array(params.disto)    
        return {"intrinsic": intrinsic, "distortion":distortion, "params": params}
        
    
    def grab_zed_frame(self,  frame_=None,  depth = True, point_cloud = False) -> dict :
        '''
        Captures an image, depth map, and point cloud from the ZED camera.

        Args:
            frame_ (str, optional)      : A string indicating the camera frame to use (left or right). Defaults to None.
            depth (bool, optional)      : Whether to capture the depth map. Defaults to True.
            point_cloud (bool, optional): Whether to capture the point cloud. Defaults to False.

        Returns:
            Union[dict, None]: A dictionary containing the image, depth map, and point cloud, or None if the grab fails.
        '''
        
        if frame_ is None:
            frame_ = self.frame_
        zed_image        = sl.Mat() 
        zed_depth_image  = sl.Mat() if depth else None
        zed_point_cloud  = sl.Mat() if point_cloud else None
        if self.zed.grab(sl.RuntimeParameters()):
            self.zed.retrieve_image(zed_image, sl.VIEW.LEFT if self.frame_=="left" else sl.VIEW.RIGHT)
            if depth:
                self.zed.retrieve_measure(zed_depth_image, sl.MEASURE.DEPTH) 
            if point_cloud:
                self.zed.retrieve_measure(zed_point_cloud, sl.MEASURE.XYZRGBA)   
            return {"rgb": zed_image, "depth": zed_depth_image, "point_cloud":zed_point_cloud}
        return None

        
    def __del__(self) -> None:
        '''
        Closes the ZED camera connection when the object is deleted.

        Returns: None
        '''
        self.zed.close()
    