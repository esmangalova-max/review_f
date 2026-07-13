# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 17:17:44 2025

@author: User
"""
import cv2
import numpy as np
import time

import lib_video as libv
import torch

import copy
#import ffmpegcv

this_order = 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


menu_filename = 'TestVideo//0102.tsv'
classes_filename = 'TestVideo//classes_back.joblib'
model_filename = 'TestVideo//fasterrcnn_resnet50_fpn_530.pth'
classvalue_filename = 'TestVideo//class_value.csv'
teamodel_name = 'TestVideo//ResNet18.pth'
cam_adress = 'rtsp://192.168.32.166:554/av0_0'

width = 1920
height = 1080

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(cam_adress)


frame_width = int((width + 1000)/2)
frame_height = int(height/2)
out = cv2.VideoWriter('test0102.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (frame_width,frame_height))
#out = ffmpegcv.VideoWriter('output.mp4', None, 25)


try:
    menu_list, menu_day = libv.make_menu(menu_filename)
except:
    print('Error. Daily Menu is not loaded')
classes_back = libv.make_classes_back(classes_filename)
  

class_value = libv.make_classes_values(classvalue_filename)

try:
    model = libv.load_model(model_filename, classes_back)
    model.to(device)
except:
    print('Error. Model is not loaded')


try:
    teamodel = libv.load_tea_model(teamodel_name)
except:
    print('Error. Tea Model is not loaded')

torch.cuda.empty_cache()


        
ec = [(255,0,0),
      (0,255,0),
      (0,0,255),
      (0,255,255),
      (255,0,255),
      (255,255,0),
      (0,155,155),
      (255,0,255),
      (155,155,0),
      (155,50,0),
      (50,155,0),
      (0,50,155),
      (0,105,155),
      (155,0,105),
      (105,155,0),
      (255,0,0),
      (0,255,0),
      (0,0,255),
      (0,255,255),
      (255,0,255),
      (255,255,0),
      (0,155,155),
      (255,0,255),
      (155,155,0),
      (155,50,0),
      (50,155,0),
      (0,50,155),
      (0,105,155),
      (155,0,105),
      (105,155,0),
      
     ]

headers = ['Экстра', 'Салаты', 'Супы', 'Горячее', 'Гарниры', 'Напитки', 'Сухари, соусы', 'Хлебобулочные изделия', 'Прочее']
    
    

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
 
iii = 0
dishes_choosed = None
dishes_prev = None
dishes_ = None
# Read until video is completed


Times = [
[(9 * 60  + 15)*25, (9*60 + 45)*25],
[(9 * 60  + 55)*25, (9*60 + 45)*25],
[(10 * 60  + 20)*25, (10*60 + 30)*25],
[(11 * 60  + 15)*25, (11*60 + 40)*25],
[(27 * 60  + 10)*25, (27*60 + 30)*25],
[(100 * 60  + 15)*25, (100*60 + 60)*25]]


def status_check(iii):
    this_order = 0
    while iii > Times[this_order][1]:
        this_order += 1
    if iii >= Times[this_order][0] and iii < Times[this_order][1]:
        return True
    return False

    iii += 1
    print(iii)
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True: 
        if True:
            t = time.time()
                
            model.eval()
            #cpu_device = torch.device("cuda")
            frame_ = libv.get_image(frame, transforms=libv.get_valid_transform())
            f = [frame_.to(device)]
                
            # Распознавание блюд нейронной сетью
            outputs = model(f)
            cpu_device = torch.device("cpu")
                
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            
            if len(outputs[0]['scores']) > 0:
                t = time.time() - t
                
                # Отбор блюд согласно дневному меню
                try:
                    dishes_choosed = libv.choose_dish(outputs, classes_back, menu_list, class_value, frame, teamodel)
                except:
                    print('Error. Dishes are not choosed') 
                dishes = libv.test_image(dishes_choosed, menu_day)
                    
                #### Добавление блюд с предыдущего кадра
                    
                if dishes_prev is not None:
                    try:
                        dishes_ = libv.compare(dishes_prev, dishes)
                    except:
                        print('Error. Dishes are not compared') 
                    dishes_prev = copy.copy(dishes_)
                else:
                    dishes_ = copy.copy(dishes)
                    dishes_prev = copy.copy(dishes_)
                    
                print(dishes)
                # Отображение результатов распознавания 
                try:
                    hl = libv.make_image(frame, dishes_, headers, ec, frame_width, frame_height)
                except:
                    print('Error. Image is not created')  
            else:
                t = time.time() - t
                    
                # Отображение результатов распознавания 
                try:
                    hl = libv.make_image_empty(frame, frame_width, frame_height)
                except:
                    print('Error. Empty image is not created')
            
            out.write(hl)


        else:
            frame_ = libv.get_image(frame, transforms=libv.get_valid_transform())
            if dishes_ is None:
                try:
                    hl = libv.make_image_empty(frame, frame_width, frame_height)
                except:
                    print('Error. Empty image is not created')
            else:
                try:
                    hl = libv.make_image(frame, dishes_, headers, ec, frame_width, frame_height)
                except:
                    print('Error. Image is not created')                    
            out.write(hl)

        dishes_choosed = None
        dishes_prev = None
        dishes_ = None
    else: 
        break

# When everything done, release the video capture object
cap.release()
out.release()