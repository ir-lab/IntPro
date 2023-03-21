


import cv2
import numpy as np

import sys
import os
from intpro_utils import intpro_utils


def contrast_control(img,alpha=1,beta=0):
    # nimg = np.zeros((img.shape),dtype=np.uint8)
    nimg = np.clip(img*alpha + beta,0, 255)
    return nimg


def resize_image(image, factor = 0.5):
    if len(image.shape) == 3:
        h,w,c = image.shape
    elif len(image.shape) == 2:
        h,w = image.shape
    else:
        return None
    return cv2.resize(image,(int(w*factor), int(h*factor)))

# modes = ["static_mode", "dynamic_mode", "dual_mode", "noproj_mode"]
modes = ["dual_mode"]

mode = sys.argv[2]+"_mode"
user = sys.argv[1]
gray = int(sys.argv[3])
data_dir = f"/share/iros_1_dataset/{user}/"

# for mode in modes:
mode_files = os.listdir(os.path.join(data_dir,mode))
rgb_file = [f for f in mode_files if f.split("_")[-2] == "rgb"]
rgb_file.sort()

global alpha, beta
alpha = 0.002
beta = 0
move_flag = True

def on_trackbar_alpha(val):
    global alpha
    alpha = float(val/100)

def on_trackbar_beta(val):
    global beta
    beta = float(val/100)
    
idx = 0
while True:
    file = rgb_file[idx]
    print(mode,file)
    file_name = os.path.join(data_dir,mode,file)
    img = np.load(file_name)
    if gray:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        

    # img = contrast_control(img,alpha=alpha,beta=beta)
    cv2.imshow("rgb",resize_image(img))
    # cv2.createTrackbar("alpha_control","rgb",0,100,on_trackbar_alpha)
    # cv2.createTrackbar("beta_control","rgb",0,100,on_trackbar_beta)
    key = cv2.waitKey(100)
    if key == ord('.'):
        idx += 1
    
    if key == ord(','):
        idx -= 1
        
    if idx == len(rgb_file):
        idx = len(rgb_file) -1

    if idx <= 0:
        idx = 0
        