

import numpy as np
import cv2
import time
import yaml
from tqdm import tqdm
import os
import sys
import json
from typing import Any, Dict
import time
from time import perf_counter
from functools import wraps
import tracemalloc
from transforms3d.affines import compose
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2mat, mat2euler
import datetime


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)




class intpro_utils:
    
    @staticmethod
    def load_yaml(filename:str) -> Any:
        assert(os.path.isfile(filename))
        try:
            with open(filename,'r') as fh:
                data = yaml.load(fh,Loader=yaml.CLoader)
        except Exception as e:
            print(e)
            return False
        return data

   
    @staticmethod
    def save_json(data: Any, file_name: str) -> bool:
        try:
            with open(file_name,"w",encoding="utf-8") as fh:
                json.dump(data,fh,cls=NpEncoder,indent=1)
        except Exception as e:
            print(e)
            return False
        return True 
    
    
    @staticmethod
    def load_json(file_name: str) -> None:
        try:
            with open(file_name,"r", encoding="utf-8") as fh:
                data  = json.loads(fh.read())
                return data
        except Exception as e:
            print(f"got error while loading json file: {e}")
            return None
    
    
    
    @staticmethod
    def resize_image(image:np.ndarray, ratio:float = 0.5):
        if image.shape == 3:
            h,w,c = image.shape
        else:
            h,w = image.shape
        
        resized_image  = cv2.resize(image, (int(h*ratio),int(w*ratio)))
        return resized_image
    
    
    @staticmethod
    def load_proj_params(filename: str, scale: float=1000.0) -> Dict:
        assert(os.path.isfile(filename))
        
        fs = cv2.FileStorage(filename,cv2.FILE_STORAGE_READ)
        
        proj_K = fs.getNode("proj_K")
        proj_K = proj_K.mat()
        proj_K = proj_K

        proj_distortion = fs.getNode("proj_kc")
        proj_distortion =proj_distortion.mat()
        
        R = fs.getNode("R")
        R = R.mat()
        T = fs.getNode("T")
        T = T.mat()
        T = T/scale
        T = T.reshape(3)

        extrinsic_mat = intpro_utils.get_compose(trans=T,rot_mat=R)
        
        params = {"intrinsic":proj_K, "distortion": proj_distortion, "rotation": R, "translation": T, "extrinsic": extrinsic_mat}
        return params
    
    
    @staticmethod
    def load_cam_params(filename: str) -> Dict:
        assert(os.path.isfile(filename))
       
        params = intpro_utils.load_yaml(filename=filename)

        cam_mat     = np.asanyarray(params["camera_matrix"]["data"])
        cam_mat     = cam_mat.reshape(params["camera_matrix"]["rows"],
                                params["camera_matrix"]["cols"])

        dist_coef   = np.asanyarray(params["distortion_coefficients"]["data"])
        dist_coef   = dist_coef.reshape(params["distortion_coefficients"]["rows"],
                                    params["distortion_coefficients"]["cols"])

        dist_model  = params["distortion_model"]

        width       = params["image_width"]
        height      = params["image_height"]


        proj_mat    = np.asanyarray(params["projection_matrix"]["data"])
        proj_mat    = proj_mat.reshape(params["projection_matrix"]["rows"],
                                    params["projection_matrix"]["cols"])

        rect_mat    = np.asanyarray(params["rectification_matrix"]["data"])
        rect_mat    = rect_mat.reshape(params["rectification_matrix"]["rows"],
                                    params["rectification_matrix"]["cols"])

        params = {"intrinsic": cam_mat, 
                  "distrotion": dist_coef, 
                  "distortion_mdel": dist_model, 
                  "width": width, "height": height,
                  "projection": proj_mat,
                  "rectification": rect_mat
                  }
        return params
    
    @staticmethod
    def get_compose(trans: None = None,
                    rot_mat: None = None,
                    quat:None = None,
                    euler: None = None,
                    zoom = [1,1,1],
                    axes = "sxyz") -> Any:
        if trans is None and quat is None:
            # assert(len(trans) == 3)
            # assert(len(quat)  == 4)
            return 

        if quat is None and euler is not None and trans is not None:
            
            return compose(T = trans,
                           R = euler2mat(*euler,axes=axes),
                           Z = zoom) 
        elif rot_mat is not None and trans is not None:
            return compose(T = trans,
                           R = rot_mat,
                           Z = zoom) 
        else:
            return compose(T = trans,
                           R = quat2mat(quat),
                           Z = zoom) 
            
            
    @staticmethod
    def project3d_2d(K:None, D:None, tf_mat:None, points:None) -> None:
        
        # if K != None and D != None and tf_mat != None and points != None:
        #     assert(tf_mat.shape[0] == 4)
        # else: 
        #     return
        rotV,_ = cv2.Rodrigues(tf_mat[0:3,0:3])
    
        points_2d = cv2.projectPoints(points, rotV, tf_mat[0:3,-1], K, D)[0]
        points_2d = np.asarray(points_2d).reshape(-1,2).astype(int)
        return points_2d

    @staticmethod
    def get_timestamp():
        time_stamp = str(datetime.datetime.now()).replace(" ","_")
        time_stamp = time_stamp.replace("-","_")
        time_stamp = time_stamp.replace(":","_")
        time_stamp = time_stamp.replace(".","_")
        return time_stamp
    
    @staticmethod
    def keyEvent(cv_key:cv2.waitKey(), 
                 trans: None = None,
                 euler: None = None,
                 tf_mat: None = None,
                 tmp_projector_intrinsic: None = None,
                 tmp_projector_distortion: None = None,
                 wait_ms:int      = 10,
                 xoffSet:float    = 0.005,
                 yoffSet:float    = 0.005,
                 zoffSet:float    = 0.005,
                 rotoffSet:float  = 0.005,
                 ext_offset:float = 2) -> None:

        if tf_mat is not None:
            trans = tf_mat[0:3,-1]
            euler = np.array(mat2euler(tf_mat[0:3,0:3])).reshape(-1)
            
        if cv_key >= 0:
            print(f"Called the cv2 waitKey: {cv_key}")
        else:
            print("Waiting for key input!")
            return

        # For translation
        if cv_key == ord('x'):
            trans[0] += xoffSet
        if cv_key == ord('z'):
            trans[0] -= xoffSet
        if cv_key == ord('d'):
            trans[1] += yoffSet
        if cv_key == ord('a'):
            trans[1] -= yoffSet
        if cv_key == ord('w'):
            trans[2] += zoffSet
        if cv_key == ord('s'):
            trans[2] -= zoffSet

        # For rotation
        if cv_key == ord('k'):
            euler[0] += rotoffSet
        if cv_key == ord('i'):
            euler[0] -= rotoffSet
        if cv_key == ord('j'):
            euler[1] += rotoffSet
        if cv_key == ord('l'):
            euler[1] -= rotoffSet
        if cv_key == ord('u'):
            euler[2] += rotoffSet
        if cv_key == ord('o'):
            euler[2] -= rotoffSet
            
        # for intrinsic of projector
        if cv_key == ord('f'):
            tmp_projector_intrinsic[0,0] -= ext_offset
        if cv_key == ord('g'):
            tmp_projector_intrinsic[0,0] += ext_offset
        if cv_key == ord('v'):
            tmp_projector_intrinsic[1,1] -= ext_offset
        if cv_key == ord('b'):
            tmp_projector_intrinsic[1,1] += ext_offset
        if cv_key == ord('1'):
            tmp_projector_intrinsic[0,2] -= ext_offset
        if cv_key == ord('2'):
            tmp_projector_intrinsic[0,2] += ext_offset
        if cv_key == ord('3'):
            tmp_projector_intrinsic[1,2] -= ext_offset
        if cv_key == ord('4'):
            tmp_projector_intrinsic[1,2] += ext_offset
        
        tf_mat[0:3,0:3] = euler2mat(euler[0],euler[1],euler[2])
        # print(euler,trans)
        return
    
    @staticmethod
    def get_plane(plane_type:str = "table", width:float    = 1,  
                                            length:float   = 1,
                                            origin_x:float = 0.0,
                                            origin_y:float = 0.0,
                                            height:float   = 0.0) -> None:
        
        n_x = (-width/2.0) + origin_x
        p_x = ( width/2.0) + origin_x
        n_y = (-length/2.0) + origin_y
        p_y = ( length/2.0) + origin_y
        
        plane_3d = np.array([[n_x,p_y,height],
                             [n_x,n_y,height],
                             [p_x,n_y,height],
                             [p_x,p_y,height]])
        return plane_3d
    
    @staticmethod
    def measure_performance(func):
        '''Meausure performace of the function'''
        @wraps(func)
        def wrapper(*args, **kwargs):
            
            tracemalloc.start()
            start_time = perf_counter()
            func(*args,**kwargs)
            current, peak = tracemalloc.get_traced_memory()
            finish_time = perf_counter()
            print(f'Function: {func.__name__}')
            print(f'Method: {func.__doc__}')
            print(f'Memory usage:\t\t {current / 10**6:.6f} MB \n'
                f'Peak memory usage:\t {peak / 10**6:.6f} MB ')
            print(f'Time elapsed is seconds: {finish_time - start_time:.6f}')
            print(f'{"-"*40}')
            tracemalloc.stop()
        return wrapper
    
    @staticmethod
    def show_projector_screen(x_offset:int, y_offset:int, image:None = None, project_name:str = "Optoma") -> None:
        cv2.namedWindow(project_name,cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(project_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(project_name,x_offset,y_offset)
        cv2.imshow(project_name,image)
    
    @staticmethod
    def edit_apriltag(tags_data_dir:str, tag_id:int, resize_factor:float = 2) -> None:
        image = cv2.imread(os.path.join(tags_data_dir,f"tag36_11_{tag_id:05}.png"))        
        image = image[200:1000,:,:]
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        size_ = image.shape[0]
        image = cv2.resize(image,(int(size_/resize_factor),int(size_/resize_factor)))
        
        # padd the tag
        size_ = image.shape[0]
        buff_offset = 50
        buff  = 0 * np.ones((size_,buff_offset))
        image = np.hstack((buff,image,buff))
        buff =  0 * np.ones((buff_offset, size_+ 2* buff_offset))
        image = np.vstack((buff,image,buff))
        buff =  255 * np.ones((buff_offset, size_+ 2* buff_offset))
        image = np.vstack((buff,image))
        cv2.putText(image,f"tag_{tag_id}",(int(image.shape[1]/2 - 10),(25)),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0),2)
        cv2.imwrite(f"tags_data/resized_tag36_11_{tag_id:05}.png",image)
        return image

if __name__ == '__main__':
        
    id = 0
    tags_data_dir = "/share/tags_data" 
    tags_file = os.listdir(tags_data_dir)
    while True:
        image = intpro_utils.edit_apriltag(tags_data_dir=tags_data_dir, tag_id=id, resize_factor=3)
        # cv2.imshow(f"tag_{id}",image)
        # cv2.waitKey(1000)
        print(id)
        id += 1
        if id == len(tags_file):
            break