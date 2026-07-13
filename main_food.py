
import cv2
import numpy as np
import time

import lib_video as libv
import torch

import copy
import os
from pathlib import Path


#import ffmpegcv

this_order = 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


menu_filename = 'TestVideo//0102.tsv'
classes_filename = 'TestVideo//classes_back.joblib'
model_filename = 'TestVideo//fasterrcnn_resnet50_fpn_530.pth'
classvalue_filename = 'TestVideo//class_value.csv'
teamodel_name = 'TestVideo//ResNet18.pth'
videofilename = 'TestVideo//Cam 8_01_02_2025 11.50.00.mp4'

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(videofile_name)

width = 1920
height = 1080

frame_width = int((width + 1000)/2)
frame_height = int(height/2)
out = cv2.VideoWriter('test0102.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (frame_width,frame_height))
#out = ffmpegcv.VideoWriter('output.mp4', None, 25)


file_obj = Path(menu_filename)

if file_obj.exists():
    menu_list, menu_day = libv.make_menu(menu_filename)
else:
    print(f"Файл {menu_filename} не найден. Создайте его или проверьте путь.")
 
 
file_obj = Path(classes_filename)

if file_obj.exists():
    classes_back = libv.make_classes_back(classes_filename)
else:
    print(f"Файл {classes_filename} не найден. Создайте его или проверьте путь.")
  
file_obj = Path(classvalue_filename)

if file_obj.exists():
    class_value = libv.make_classes_values(classvalue_filename)
else:
    print(f"Файл {classvalue_filename} не найден. Создайте его или проверьте путь.")

file_obj = Path(model_filename)

if file_obj.exists():
    model = libv.load_model(model_filename, classes_back)
    model.to(device)
else:
    print(f"Файл {model_filename} не найден. Создайте его или проверьте путь.")



file_obj = Path(teamodel_name)

if file_obj.exists():
    teamodel = libv.load_tea_model(teamodel_name)
else:
    print(f"Файл {teamodel_name} не найден. Создайте его или проверьте путь.")




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
 
 
stay = 1
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

def nn_check(stay):
    if stay <= 0:
        return True
    else:
        return False


while(cap.isOpened()):
    iii += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    if iii%1000 == 0:
        print(iii)
    if ret == True: 
        if status_check(iii):
            
            stay -= 1
        
            if nn_check(stay):
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
                    dishes_choosed = libv.choose_dish(outputs, classes_back, menu_list, class_value, frame, teamodel)
                    dishes = libv.test_image(dishes_choosed, menu_day)
                    
                    #### Добавление блюд с предыдущего кадра
                    
                    if dishes_prev is not None:
                        dishes_ = libv.compare(dishes_prev, dishes)
                        dishes_prev = copy.copy(dishes_)
                    else:
                        dishes_ = copy.copy(dishes)
                        dishes_prev = copy.copy(dishes_)
                    
                    print(dishes)
                    # Отображение результатов распознавания 
                    hl = libv.make_image(frame, dishes_, headers, ec, frame_width, frame_height)

                else:
                    t = time.time() - t
                    
                    # Отображение результатов распознавания 
                    hl = libv.make_image_empty(frame, frame_width, frame_height)
            
                stay = np.floor(t*25)
                out.write(hl)


            else:
                ### Только отображение результатов предыдущего распознавания
                if iii%5 == 0:
                
                    frame_ = libv.get_image(frame, transforms=libv.get_valid_transform())
                    if dishes_ is None:
                        hl = libv.make_image_empty(frame, frame_width, frame_height)
                    else:
                        hl = libv.make_image(frame, dishes_, headers, ec, frame_width, frame_height)
                                    
                    out.write(hl)

        else:
            dishes_choosed = None
            dishes_prev = None
            dishes_ = None
    else: 
        break

# When everything done, release the video capture object
cap.release()
out.release()
