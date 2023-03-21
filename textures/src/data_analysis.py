
from calendar import c
import enum
import os
from pyexpat import model
from re import A
import sys
from tkinter import filedialog
import argparse
import numpy as np
import cv2
# import pyopenpose as op
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from intpro_utils import intpro_utils
import threading

class DATA_ANALYSIS:
    
    def __init__(self, data_dir = "/share/icra_dataset", model_folder = "/home/slocal/github/openpose/models/", allow_collision=False, freq:int = 10) -> None:
        self.package_path = os.path.dirname(os.path.abspath(os.path.join(__file__,"..")))
        self.data_dir = data_dir
        # self.modes = ["c_mode","d_mode","dual_mode","noproj_mode"]    
        self.modes = ["static_mode","dynamic_mode","dual_mode","noproj_mode"]    
        self.perf_metrics = ["eng", "exc", "str", "rel", "int", "foc"]
        dirs = os.listdir(self.data_dir)
        self.users = [user for user in dirs if len(user.split("_")) == 1]
        
        params = dict()
        params["model_folder"] = model_folder
        # Starting OpenPose
        # self.opWrapper = op.WrapperPython()
        # self.opWrapper.configure(params)
        # self.opWrapper.start()
        # # Process Image
        # self.datum = op.Datum()
        
        self.all_meta_data = []
        self.allow_collision = allow_collision
        self.freq = freq
        self.visual_img = None
        self.alpha = 100
        self.beta  = 200
        self.brightness = 255
        self.contrast = 127
        self.th = threading.Thread(target=self.test,args=())
        self.th_start = True
        
    def analyze_metdata(self, met_files, user, mode = "c_mode"):
        user_dir = os.path.join(self.data_dir,user,mode)
        met_data = []
        print(f"filtering performance metrics files..... for {user} and {mode}")
        for f in tqdm(met_files):
            data = np.load(os.path.join(user_dir,f))
            
            # if data != []:
            if len(data) > 0:
                data_ = [data[1], data[3], data[6], data[8], data[10],data[12]]

                filtered_data = []
                for d in data_:
                    if d == -1:
                        filtered_data.append(0.0)
                    else:
                        filtered_data.append(d)
                        
                met_data.append(filtered_data)
        met_data = np.array(met_data)
        met_data = np.unique(met_data,axis=0)
        avg_met_data = np.average(met_data,axis=0)
        # fig = plt.figure(f"{mode}")
        # for idx in range(6):
        #     ax = fig.add_subplot(6,1,idx+1)
        #     x = [i for i in range(met_data.shape[0])]
        #     y = met_data[:,idx]
        #     ax.bar(x,y)
        # ax = fig.add_subplot(1,1,1)
        # x = [i for i in range(6)]
        # y = avg_met_data
        # bars = ax.bar(x,np.round(y,decimals=2))
        # ax.set_xticks(x)
        # ax.set_xticklabels(self.perf_metrics)
        # ax.set_ylim(0,1)        
        # ax.bar_label(bars)
        # fig.savefig(os.path.join(f"{mode}.pdf"))
        return met_data, avg_met_data
    
    def analyze_all_metdata(self):
       
        for user in self.users:
            c_mode_avg_metdata = []
            d_mode_avg_metdata = []
            dual_mode_avg_metdata = []
            noproj_mode_avg_metdata = []
            if user == "user_xiao" or user == "user_test":
                continue
            for mode in self.modes:
                output = self.filter_userdata(user=user, mode=mode)
                met_data,avg_metdata = self.analyze_metdata(met_files=output["met_files"], 
                                                            user=user,
                                                            mode=mode)    
                if mode == "c_mode":
                    c_mode_avg_metdata.append(avg_metdata)
                elif mode == "d_mode":
                    d_mode_avg_metdata.append(avg_metdata)
                elif mode == "dual_mode":
                    dual_mode_avg_metdata.append(avg_metdata)
                elif mode == "norproj_mode":
                    noproj_mode_avg_metdata.append(avg_metdata)
                    
                    
            np.save(os.path.join("/share/icra_dataset",user,mode,"filtered_c_mode_metdata"),c_mode_avg_metdata)
            np.save(os.path.join("/share/icra_dataset",user,mode,"filtered_d_mode_metdata"),d_mode_avg_metdata)
            np.save(os.path.join("/share/icra_dataset",user,mode,"filtered_dual_mode_metdata"),dual_mode_avg_metdata)
            np.save(os.path.join("/share/icra_dataset",user,mode,"filtered_noproj_mode_metdata"),c_mode_avg_metdata)
        # c_mode_avg_metdata      = np.average(np.array(c_mode_avg_metdata)     ,axis=0)
        # d_mode_avg_metdata      = np.average(np.array(d_mode_avg_metdata)     ,axis=0)
        # dual_mode_avg_metdata   = np.average(np.array(dual_mode_avg_metdata),axis=0)
        # noproj_mode_avg_metdata = np.average(np.array(noproj_mode_avg_metdata),axis=0)
        # x = [i for i in range(6)]
        # self.plot_bar(x = x , y = c_mode_avg_metdata, plot_name =  "c_mode_avg_metdata")
        # self.plot_bar(x = x , y = d_mode_avg_metdata, plot_name =  "d_mode_avg_metdata")
        # self.plot_bar(x = x , y = dual_mode_avg_metdata, plot_name =  "dual_mode_avg_metdata")
        # self.plot_bar(x = x , y = noproj_mode_avg_metdata, plot_name =  "noproj_mode_avg_metdata")
        
                    
    def plot_bar(self,x,y,plot_name,savefig = True):
        c_fig = plt.figure(plot_name)
        ax = c_fig.add_subplot(1,1,1)
        bars = ax.bar(x,np.round(y,decimals=2))
        ax.set_xticks(x)
        ax.set_xticklabels(self.perf_metrics)
        ax.set_ylim(0,1)        
        ax.bar_label(bars)
        if savefig:     
            c_fig.savefig(os.path.join(self.package_path,"plots",f"{plot_name}.pdf"))
        
             
    def get_time(self, filename):
        tokens = filename.split('_')[:7]
        tokens = [int(x) for x in tokens]
        time = datetime.datetime(*tokens)
        return time

    def annotate_openpose_data(self, rgb_files, depth_files, user, mode):
        user_dir = os.path.join(self.data_dir,user)
        mode_dir = os.path.join(user_dir,mode)

        rgb_files = rgb_files[::10]
        depth_files = depth_files[::10]
        len_seq = len(rgb_files)
        print(f'{len_seq} images for annotation in this sequence')

        pose_keypoints = [None] * len_seq # a list of (seq_len * n * 25 * 3), where n is the number of skeleton candidates
        selected_pose_idx = [0] * len_seq

        rp_data = [0] * len_seq
        rs_data = [0] * len_seq
        hp_data = [0] * len_seq
        go_data = [0] * len_seq
        ss_data = [0] * len_seq
        mask = [1] * len_seq
        time_stamp = [0] * len_seq
        file_id = [''] * len_seq


        init_time = self.get_time(rgb_files[0])
        flag_save = False
        save_path = ''

        curr_idx = 0
        while True:
            # robot_stop = False
            # robot_pick = False
            # human_pick = False
            # shadow_stop = False
            
            
            # image = np.load(os.path.join(mode_dir,f))
            # image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
            # image = cv2.resize(image, (int(image.shape[1]variable

            curr_img_file = rgb_files[curr_idx]
            curr_depth_file = depth_files[curr_idx]
            
            if file_id[curr_idx] == '':
                file_id[curr_idx] = '_'.join(curr_img_file.split('_')[:7])
                current_time = self.get_time(curr_img_file)
                time_diff = (current_time - init_time).total_seconds()
                time_stamp[curr_idx] = time_diff

            try:
                image = np.load(os.path.join(mode_dir,curr_img_file))
                image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
                # image = cv2.resize(image, (int(image.shape[1]/2),int(image.shape[0]/2)))
                # self.datum.cvInputData = image
                # self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
                # curr_keypoints_candidates = self.datum.poseKeypoints

                depth = np.nan_to_num(np.load(os.path.join(mode_dir,curr_depth_file)))
                
                # Process for each of the skeleton candidate
                # processed_keypoints_candidates = []
                # for candidate_idx in range(curr_keypoints_candidates.shape[0]):
                #     candidate_keypoints = curr_keypoints_candidates[candidate_idx]

                #     # for the current skeleton candidate, get the depth of key points
                #     processed_keypoints = []
                #     for keypoint_i in range(candidate_keypoints.shape[0]):
                #         key_point = candidate_keypoints[keypoint_i]
                #         if key_point[2] > 0.2:
                #             depth_point = depth[round(key_point[1]), round(key_point[0])]
                #         else:
                #             depth_point = 0
                #         processed_keypoint = [key_point[0], key_point[1], key_point[2], depth_point]
                #         processed_keypoints.append(processed_keypoint)

                #     processed_keypoints_candidates.append(processed_keypoints)

                # curr_keypoints_candidates = processed_keypoints_candidates
                # print(np.array(curr_keypoints_candidates).shape)
                # pose_keypoints[curr_idx] = curr_keypoints_candidates

                
            except Exception as e:
                print(f"Got error in open pose annotation: {e}")
                continue
            
            # print(curr_keypoints, type(curr_keypoints), curr_keypoints.shape)
            # if pose_keypoints[curr_idx] is None:
            #     pose_keypoints[curr_idx] = curr_keypoints
            	
            # output_image1 = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            # output_image1 = output_image1 // 5 * 2
            output_image1 = image
            output_image = np.zeros(output_image1.shape,dtype=np.uint8)
            # for candidate_idx in range(len(curr_keypoints_candidates)):
            #     for keypoint_idx in range(len(curr_keypoints_candidates[candidate_idx])):
            #         keypoint = curr_keypoints_candidates[candidate_idx][keypoint_idx][:2]
            #         if candidate_idx == selected_pose_idx[curr_idx]:
            #             color = (96, 96, 255)
            #         else:
            #             color = (128, 128, 128)
            #         output_image = cv2.circle(output_image, (round(keypoint[0]), round(keypoint[1])), 10, color, -1)
            #         output_image = cv2.putText(output_image, f'{candidate_idx}', (round(keypoint[0])-3, round(keypoint[1])+3), cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 0), 2)
            if hp_data[curr_idx] == 1:
                output_image = cv2.rectangle(output_image, (20,10), (240,40), (144, 144, 222), -1)
            if rp_data[curr_idx] == 1:
                output_image = cv2.rectangle(output_image, (20,50), (240,80), (144, 144, 222), -1)
            if ss_data[curr_idx] == 1:
                output_image = cv2.rectangle(output_image, (20,90), (240,120), (144, 144, 222), -1)
            if go_data[curr_idx] == 1:
                output_image = cv2.rectangle(output_image, (20,130), (240,160), (144, 144, 222), -1)
            if rs_data[curr_idx] == 1:
                output_image = cv2.rectangle(output_image, (20,170), (240,200), (144, 144, 222), -1)
            if mask[curr_idx] == 1:
                output_image = cv2.rectangle(output_image, (20,210), (240,240), (144, 144, 222), -1)
            if flag_save == True:
                output_image = cv2.putText(output_image,f"anotation saved at {save_path}",  (20,210),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                flag_save = False
                save_path = ''
            output_image = cv2.putText(output_image,"human_pick: {} (press: 1/q)".format(hp_data[curr_idx]),  (20,40) ,cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            output_image = cv2.putText(output_image,"robot_pick: {} (press: 2/w)".format(rp_data[curr_idx]),  (20,80) ,cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)
            output_image = cv2.putText(output_image,"shadow_stop: {} (press: 3/e)".format(ss_data[curr_idx]), (20,120),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)
            output_image = cv2.putText(output_image,"highlight_on: {} (press: 4/r)".format(go_data[curr_idx]),(20,160),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)
            output_image = cv2.putText(output_image,"robot_stop: {} (press: 5/t)".format(rs_data[curr_idx]),  (20,200),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2) 
            output_image = cv2.putText(output_image,"mask: {} (press: a)".format(mask[curr_idx]),             (20,240),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  
            output_image = cv2.putText(output_image,"prev: press ,",                                          (20,280),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2) 
            output_image = cv2.putText(output_image,"next: press .",                                          (20,320),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)   
            output_image = cv2.putText(output_image,"switch keypoints: press t",                              (20,360),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  
            output_image = cv2.putText(output_image,"next sequence: press n",                                 (20,400),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  
            output_image = cv2.putText(output_image,f"mode: {mode}",                                          (20,440),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  

            output_image = cv2.putText(output_image,f"file_id: {file_id[curr_idx]}",  (600,40),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  
            output_image = cv2.putText(output_image,f"{curr_idx}/{len_seq}, time_stamp: {time_stamp[curr_idx]:.02f}",  (600,80),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  
            output_image = cv2.putText(output_image,f"total_rp = {sum(rp_data[:curr_idx])}",  (600,120),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  
            output_image = cv2.putText(output_image,f"total_rs = {sum(rs_data[:curr_idx])}",  (600,160),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  
            output_image = cv2.putText(output_image,f"total_hp = {sum(hp_data[:curr_idx])}",  (600,200),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  
            output_image = cv2.putText(output_image,f"total_ss = {sum(ss_data[:curr_idx])}",  (600,240),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  
            output_image = cv2.putText(output_image,f"total_go = {sum(go_data[:curr_idx])}",  (600,280),cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)  

            cv2.imshow("Annotation_info", output_image)
            cv2.imshow("Visual", output_image1)
            self.visual_img = np.copy(output_image1)
            # edges = cv2.Canny(cv2.cvtColor(self.visual_img,cv2.COLOR_BGR2GRAY),self.alpha,self.beta)
            # # cv2.namedWindow("edges")
            # cv2.imshow("edges",edges)
            # cv2.createTrackbar("alpha","edges",0,1000,self.edge_tracking)    
           
            # if self.th_start:
            #     # cv2.createTrackbar("beta","edges",0,1000,self.edge_tracking)    
            #     cv2.createTrackbar('Brightness','Visual', 255, 2 * 255, self.brightness_contrast) 
                
            #     # Contrast range -127 to 127
            #     cv2.createTrackbar('Contrast', 'Visual', 127, 2 * 127,  self.brightness_contrast)  
            #     self.th_start = False
            # self.visual_img = self.controller(self.visual_img, self.brightness, self.contrast)
            
            # cv2.imshow("Visual2", self.visual_img)
            
            # self.brightness_contrast(0)
            
            key = cv2.waitKey(-1)
            if key == ord('1'):
                hp_data[curr_idx] = 1
                if not self.allow_collision:
                    rp_data[curr_idx] = 0
                    ss_data[curr_idx] = 0
                    rs_data[curr_idx] = 0
                for idx in range(curr_idx+1):
                    mask[idx] = 1
            
            elif key == ord('2'):
                rp_data[curr_idx] = 1
                if not self.allow_collision:
                    hp_data[curr_idx] = 0
                    ss_data[curr_idx] = 0
                    rs_data[curr_idx] = 0
                    go_data[curr_idx] = 0
                for idx in range(curr_idx+1):
                    mask[idx] = 1
            
            elif key == ord('3'):
                ss_data[curr_idx] = 1
                if not self.allow_collision:
                    hp_data[curr_idx] = 0
                    rp_data[curr_idx] = 0
                    rs_data[curr_idx] = 0
                    go_data[curr_idx] = 0
                for idx in range(curr_idx+1):
                    mask[idx] = 1
            
            elif key == ord('4'):
                go_data[curr_idx] = 1
                if not self.allow_collision:
                    hp_data[curr_idx] = 0
                    rp_data[curr_idx] = 0
                    rs_data[curr_idx] = 0
                for idx in range(curr_idx+1):
                    mask[idx] = 1
                    
                    
            elif key == ord('5'):
                rs_data[curr_idx] = 1
                if not self.allow_collision:
                    hp_data[curr_idx] = 0
                    rp_data[curr_idx] = 0
                    ss_data[curr_idx] = 0
                    go_data[curr_idx] = 0
                for idx in range(curr_idx+1):
                    mask[idx] = 1

            if key == ord('q'):
                hp_data[curr_idx] = 0
            
            elif key == ord('w'):
                rp_data[curr_idx] = 0
            
            elif key == ord('e'):
                ss_data[curr_idx] = 0
            
            elif key == ord('r'):
                go_data[curr_idx] = 0
            
            elif key == ord('t'):
                rs_data[curr_idx] = 0
                
                # selected_pose_idx[curr_idx] = (selected_pose_idx[curr_idx] + 1) % len(curr_keypoints_candidates)
            elif key == ord('a'):
                for idx in range(curr_idx+1):
                    mask[idx] = 1
                for idx in range(curr_idx, len_seq):
                    mask[idx] = 0

            elif key == ord(','):
                curr_idx = max(0, curr_idx - 1)

            elif key == ord('.'):
                curr_idx = min(curr_idx + 1, len_seq - 1)
            
            elif key == ord('s'):
                to_save = {}
                to_save['rp_data'] = rp_data
                to_save['rs_data'] = rs_data
                to_save['hp_data'] = hp_data
                to_save['ss_data'] = ss_data
                to_save['go_data'] = go_data
                
                to_save['mask'] = mask
                to_save['time_stamp'] = time_stamp
                to_save['file_id'] = file_id
                for i in range(len_seq):
                    pose_keypoints[i] = pose_keypoints[selected_pose_idx[i]]
                # to_save['pose_keypoints'] = pose_keypoints
                to_save['pose_keypoints'] = None

                
                annotation_dir = os.path.join(self.data_dir,"users_annotations",user)
                if not os.path.isdir(annotation_dir):
                    os.mkdir(annotation_dir)
                    
                if not allow_collision:
                    save_path = os.path.join(annotation_dir,f'{mode}.json')
                else:
                    save_path = os.path.join(annotation_dir,f'{mode}_allow_collision.json')
                try:
                    intpro_utils.save_json(to_save, save_path)
                    flag_save = True
                except e:
                    print(e)
            
            elif key == ord('n'):
                return
            
            # # o1    
            # print("waiting for second key")
            # key = cv2.waitKey(-1)
            
            # print("data appended")
            # rs_data.append(robot_stop)
            # ss_data.append(shadow_stop)
            # hp_data.append(human_pick)
            # rp_data.append(robot_pick)
    
    def test(self):
        while True:
            if self.visual_img is not None:
                print("running")
                # edges = cv2.Canny(cv2.cvtColor(self.visual_img,cv2.COLOR_BGR2GRAY),self.alpha,self.beta)
                # cv2.namedWindow("edges")
                # cv2.imshow("edges",edges)
                # cv2.createTrackbar("alpha","edges",0,1000,self.edge_tracking)    
                # cv2.createTrackbar("beta","edges",0,1000,self.edge_tracking)   
                cv2.imshow("filter",self.visual_img)
    
    def edge_tracking(self,val):
        self.alpha = cv2.getTrackbarPos("alpha","edges")
        self.beta = cv2.getTrackbarPos("alpha","edges")
        
    def brightness_contrast(self,brightness=0):
        print("on track start")
        self.brightness = cv2.getTrackbarPos('Brightness', 'Visual')
        self.contrast   = cv2.getTrackbarPos('Contrast','Visual')
        print(self.brightness,self.contrast)
        # cv2.setTrackbarPos('Brightness', 'Visual', self.brightness)
        # cv2.setTrackbarPos('Contrast', 'Visual', self.contrast)
        cal = self.controller()
        cv2.imshow('Effect', cal)
    
    def controller(self):
        brightness = int((self.brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    
        contrast = int((self.contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    
        if brightness != 0:
    
            if brightness > 0:
    
                shadow = brightness
    
                max = 255
    
            else:
    
                shadow = 0
                max = 255 + brightness
    
            al_pha = (max - shadow) / 255
            ga_mma = shadow
    
            # The function addWeighted calculates
            # the weighted sum of two arrays
            cal = cv2.addWeighted(self.visual_img, al_pha, 
                                  self.visual_img, 0, ga_mma)
    
        else:
            cal = self.visual_img
    
        if contrast != 0:
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)
    
            # The function addWeighted calculates
            # the weighted sum of two arrays
            cal = cv2.addWeighted(cal, Alpha, 
                                cal, 0, Gamma)
    
        # putText renders the specified text string in the image.
        cv2.putText(cal, 'B:{}, C:{}'.format(brightness,
                                            contrast), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
        # self.visual_img = cal
        return cal
        

    def filter_userdata(self,user,mode = "c_mode"):
        user_dir = os.path.join(self.data_dir,user)
        mode_dir = os.path.join(user_dir,mode)
        mode_files = os.listdir(mode_dir)
        mode_files = np.sort(mode_files)
        rgb_files   = []
        depth_files = []
        met_files   = []
        lh_files    = []
        rh_files    = []
        ee_files    = []
        for f in mode_files:
            tmp_ = f.split("_")
            if "rgb" in tmp_:
                rgb_files.append(f)
            elif "depth" in tmp_:
                depth_files.append(f)
            elif "met" in tmp_:
                met_files.append(f)
            elif "rh" in tmp_:
                rh_files.append(f)
            elif "lh" in tmp_:
                lh_files.append(f)
            elif "ee" in tmp_:
                ee_files.append(f)
            
        return {"rgb_files":rgb_files, 
                "depth_files": depth_files,
                "met_files":met_files, 
                "rh_files":rh_files,
                "lh_files": lh_files, 
                "ee_files":ee_files}

    
    def annotate_data(self):
        print("annotate start....")
        annotation_dir = os.path.join(self.data_dir,"users_annotations")
        already_annotated = os.listdir(annotation_dir)
        for user in self.users:
            if user in already_annotated and len(os.listdir(os.path.join(annotation_dir,user))) == 4:
                continue
            print(user)
            # if user != "yifan":
            #     continue
            for mode in self.modes:
                sorted_datafiles = self.filter_userdata(user=user, mode=mode)
                
                self.annotate_openpose_data(rgb_files = sorted_datafiles["rgb_files"],
                                            depth_files= sorted_datafiles["depth_files"],
                                            user = user, 
                                            mode = mode)


if  __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str,default="/data",help="Path to data directory")
    parser.add_argument("--openpose_model","-opm", type=str, default="", help="Path to the model folder")
    parser.add_argument("--freq","-f", type=int, default=10, help="Subsampling frequency....")
    args = parser.parse_args()
    # data_dir = "/share/icra_dataset"
    # model_folder = "/home/slocal/github/openpose/models"
    
    data_dir = args.data
    model_folder = args.openpose_model
    allow_collision = True
    
    data_ana = DATA_ANALYSIS(data_dir, model_folder, allow_collision, freq=args.freq)
    # t = threading.Thread(target=data_ana.test,args=())
    # t.start()
    data_ana.annotate_data()
    
    # users = data_ana.users
    # modes = data_ana.modes

    # users = ["user_kamalesh"]
    # # modes = ['c_mode']
    # for user in users:
    #     for mode in modes:
    #         sorted_datafiles = data_ana.filter_userdata(user=user, mode=mode)
    #         # data_ana.analyze_metdata(sorted_datafiles["met_files"], user=user, mode=mode)
    #         data_ana.annotate_openpose_data(rgb_files = sorted_datafiles["rgb_files"],
    #                                         depth_files= sorted_datafiles["depth_files"],
    #                                         user = user, 
    #                                         mode = mode)
    
