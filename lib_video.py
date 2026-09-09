# -*- coding: utf-8 -*-

import os
import numpy as np
from numpy import ndarray
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import copy
import joblib
import torchvision
import torch
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor, Resize
from torchvision import transforms

import matplotlib.pyplot as plt


transform = transforms.Compose([
    ToTensor(),
    Resize((300,300))
])

def count_diff(frame: ndarray, 
               frame_back: ndarray) -> int:
    """
    

    Parameters
    ----------
    frame : 
        текущий фрейм
    frame_back : 
        предыдущий фрейм

    Returns
    -------
    d : int
        мера близости фреймов

    """
    a = np.abs(frame[:, :, :] - frame_back[:, :, :])
    a1 = np.sum(a, axis = 2)
    d = np.argwhere(a1 > 110).shape[0]
    return d

def get_image(image: ndarray, 
              transforms=None : Callable[[ndarray], ndarray]) -> ndarray:
    """
    

    Parameters
    ----------
    image : cv2 image
        Изображение
    transforms : def, optional
        Преобразование изображения. The default is None.

    Returns
    -------
    image : TYPE
       Tensor изображения

    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0

        
    if transforms:
        sample = {
                    'image': image,
                    
                }
        sample = transforms(**sample)
        image = sample['image']

    return image



def get_valid_transform():
    """
    Преобразование изображения

    """
    return A.Compose([
        ToTensorV2(p=1.0)
    ])



def overfit(box1: ndarray, 
            box2: ndarray) -> int:
    """
    Доля пересечания рамок блюд

    Parameters
    ----------
    box1 : np.array(4)
        Рамка первого блюда
    box2 : np.array(4)
        Рамка второго блюда

    Returns
    -------
    float
        Доля площади пересечения рамок к наименьшей рамке

    """
	
	
	try:
        a = min(box1[2], box2[2]) - max(box1[0], box2[0])
        b = min(box1[3], box2[3]) - max(box1[1], box2[1])
        s1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        s2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        if a > 0 and b > 0:
            s = a*b/min(s1,s2)
            return s
        
		return 0
		
	except Exception as e:
	    return 0

def no_condition(dishes_choosed, n, n2, category_index, used):
    if n['in'] == 0:
        used = True
    return dishes_choosed, used 
        
def soup_condition(dishes_choosed, n, n2, category_index, used):
    if n2['in'] == 0 and n['in'] == 1:
        dishes_choosed[category_index].remove(n2)
        dishes_choosed[category_index].append(n)
    used = True
    return dishes_choosed, used 
    
def salsd_condition(dishes_choosed, n, n2, category_index, used):
    if n2['in'] == 0 and n['in'] == 1:
        dishes_choosed[category_index].remove(n2)
        dishes_choosed[category_index].append(n)
    elif n['in'] == 1 and n2['in'] == 1:
        if 'class_2' not in n:
            n2['class_2'] = n['class']
    used = True
    return dishes_choosed, used 
    
def main_condition(dishes_choosed, n, n2, category_index, used):
    if n2['in'] == 0 and n['in'] == 1:
        dishes_choosed[dishes_choosed].remove(n2)
        dishes_choosed[dishes_choosed].append(n)
    elif n['in'] == 1 and n2['in'] == 1 and n2['class'] != n['class']:
        if 'class_2' not in n2:
            n2['class_2'] = n['class']
    used = True
    return dishes_choosed, used   

def dress_condition(dishes_choosed, n, n2, category_index, used):
    if n2['in'] == 0 and n['in'] == 1:
        dishes_choosed[4].remove(n2)
        dishes_choosed[4].append(n)
        used = True
    if n['in'] == 0 and n2['in'] == 1:
        used = True
    return dishes_choosed, used     
    
 def drink_condition(dishes_choosed, n, n2, category_index, used):
    if n2['in'] == 0 and n['in'] == 1 and n['class'] < 900:
        dishes_choosed[5].remove(n2)
        dishes_choosed[5].append(n)
        used = True
    elif n2['class'] == 802 and n['class'] == 901:
        used = True
    elif n['class'] < 900:
        used = True
    return dishes_choosed, used   
        
def process_category(dishes, dishes_choosed, category_index, threshold, conditions, extra_conditions):
    for n in dishes[category_index]:
        used = False
        for n2 in dishes_choosed[category_index]:
            s = overfit(n['box'], n2['box'])
            if s > threshold and not used:
                # Обработка условий
                dishes_choosed, used = conditions(dishes_choosed, n, n2, category_index, used)
                if extra_conditions == 1:
                    if s > 0.2:
                        if n['class'] == n2['class']:
                            used = True
                
        if not used:
            dishes_choosed[category_index].append(n)
    return dishes_choosed 


def choose_dish(outputs_n, 
                classes_back, 
				menu_list, 
				class_value, 
				image, 
				teamodel):
    """
    Выбор распознанных блюд согласно дневному меню при пересечении рамок

    Parameters
    ----------
    outputs : dict
        Прогнозы сети: keys: boxes - рамки блюд (list), labels - метки классов (list), scores - вероятность принадлежности к классу (list)
    classes_back : dict
        Преобразование классов в тип блюда (кожировка групп блюда)
    menu_list : pd.DataFrame
        Дневное меню

    Returns
    -------
    dishes_choosed : dict
        Распознанные блюда по категориям

    """
		
    menu_day = {}
    for i in range(1, 9):
        menu_day[i] = menu_list[menu_list['type'] == i]
        
    it = 0

    dishes = {}
    dishes_choosed = {}
    for i in range(0, 9):
        dishes[i] = []
        dishes_choosed[i] = []

    while outputs_n['scores'][it].detach().numpy() > class_value[int(outputs_n['labels'][it].numpy())] and it < len(outputs_n['boxes']) - 1:
        box = outputs_n['boxes'][it].detach().numpy().astype(int)
        if box[3] > 220 and box[2] > 600 and box[1] < 980 and box[0] < 1700:
            class_this = classes_back[int(outputs_n['labels'][it].numpy())]

            if class_this == [904, 705, 706, 1602, 1603]:
                dish = {}
                dish['class'] = class_this
                dish['box'] = outputs_n['boxes'][it].detach().numpy().astype(int)
                dish['scores'] = outputs_n['scores'][it].detach().numpy()
                if class_this in menu_list.values:
                    dish['in'] = 1
                else:
                    dish['in'] = 0
                dish['drop'] = 0

                dishes[8].append(dish)
                    
            ### Добавки к блюдам (яйцо, помидор)
            if class_this in [414, 411]:
                dish = {}
                dish['class'] = class_this
                dish['box'] = outputs_n['boxes'][it].detach().numpy().astype(int)
                dish['scores'] = outputs_n['scores'][it].detach().numpy()
                if class_this in menu_list.values:
                    dish['in'] = 1
                else:
                    dish['in'] = 0
                dish['drop'] = 0

                dishes[0].append(dish)

            ### Соусы
            elif class_this == 700 or (class_this >= 1000 and class_this <= 1010):
                dish = {}
                dish['class'] = class_this
                dish['box'] = outputs_n['boxes'][it].detach().numpy().astype(int)
                dish['scores'] = outputs_n['scores'][it].detach().numpy()
                if class_this in menu_list.values:
                    dish['in'] = 1
                else:
                    dish['in'] = 0
                dish['drop'] = 0

                dishes[6].append(dish)
                    
            ### Супы
            elif class_this < 200:
                dish = {}
                dish['class'] = class_this
                dish['box'] = outputs_n['boxes'][it].detach().numpy().astype(int)
                dish['scores'] = outputs_n['scores'][it].detach().numpy()
                if class_this in menu_day[2].values:
                    dish['in'] = 1
                else:
                    dish['in'] = 0
                dish['drop'] = 0

                dishes[2].append(dish)
                    
                    
            ### Горячее
            elif class_this < 300 or class_this == 315:
                dish = {}
                dish['class'] = class_this
                dish['box'] = outputs_n['boxes'][it].detach().numpy().astype(int)
                dish['scores'] = outputs_n['scores'][it].detach().numpy()
                if class_this in menu_day[3].values:
                    dish['in'] = 1
                else:
                    dish['in'] = 0
                dish['drop'] = 0

                dishes[3].append(dish)
                    
            ### Гарнир
            elif class_this < 400:
                dish = {}
                dish['class'] = class_this
                dish['box'] = outputs_n['boxes'][it].detach().numpy().astype(int)
                dish['scores'] = outputs_n['scores'][it].detach().numpy()
                if class_this in menu_day[4].values:
                    dish['in'] = 1
                else:
                    dish['in'] = 0
                dish['drop'] = 0

                dishes[4].append(dish)

            ### Салаты, сало
            elif class_this < 500 or class_this == 701:
                dish = {}
                dish['class'] = class_this
                dish['box'] = outputs_n['boxes'][it].detach().numpy().astype(int)
                dish['scores'] = outputs_n['scores'][it].detach().numpy()
                if class_this in menu_day[1].values:
                    dish['in'] = 1
                else:
                    dish['in'] = 0
                dish['drop'] = 0

                dishes[1].append(dish)

            ### Булки
            elif class_this < 700:
                dish = {}
                dish['class'] = class_this
                dish['box'] = outputs_n['boxes'][it].detach().numpy().astype(int)
                dish['scores'] = outputs_n['scores'][it].detach().numpy()
                if class_this in menu_day[7].values:
                    dish['in'] = 1
                else:
                    dish['in'] = 0
                dish['drop'] = 0

                dishes[7].append(dish)
                    
            ### Напитки
            elif class_this < 1000:
                if class_this == 901:
                    box = outputs_n['boxes'][it].detach().numpy().astype(int)
                    image = image[box[1]:box[3], box[0]:box[2], :]
                    #teamodel.to(device)
                    outputs_ = test_tea(teamodel, image)
                        
                    if outputs_:
                        dish = {}
                        dish['class'] = class_this
                        dish['box'] = outputs_n['boxes'][it].detach().numpy().astype(int)
                        dish['scores'] = outputs_n['scores'][it].detach().numpy()
                        if class_this in menu_day[5].values:
                            dish['in'] = 1
                        else:
                            dish['in'] = 0
                        dish['drop'] = 0

                        dishes[5].append(dish)

                        
                else:
                    
                    dish = {}
                    dish['class'] = class_this
                    dish['box'] = outputs_n['boxes'][it].detach().numpy().astype(int)
                    dish['scores'] = outputs_n['scores'][it].detach().numpy()
                    if class_this in menu_day[5].values:
                        dish['in'] = 1
                    else:
                        dish['in'] = 0
                    dish['drop'] = 0

                    dishes[5].append(dish)

                
        it += 1  
    
    
    # Салаты
    dishes_choosed = process_category(dishes, dishes_choosed, 1, 0.5, salad_conditions, -1)
                
    # Супы
    dishes_choosed = process_category(dishes, dishes_choosed, 2, 0.5, soup_conditions, -1)
                                
    # Прочее    
    dishes_choosed = process_category(dishes, dishes_choosed, 8, 0.5, soup_conditions, -1)
    
    # Горячее
    dishes_choosed = process_category(dishes, dishes_choosed, 3, 0.8, main_conditions, -1)
 
    # Гарниры
    dishes_choosed = process_category(dishes, dishes_choosed, 4, 0.6, dress_conditions, 1)
                
    # Напитки
    dishes_choosed = process_category(dishes, dishes_choosed, 5, 0.5, drink_conditions, -1)
                
    # Булки
    dishes_choosed = process_category(dishes, dishes_choosed, 7, 0.6, no_conditions, -1)
            
    # Сухари, соусы
    for n in dishes[6]:
        if n['class'] == 1000:
            used = False
            for n2 in dishes_choosed[1]:
                s = overfit(n['box'], n2['box'])
                if s > 0.6:
                    used = True
            for n2 in dishes_choosed[7]:
                s = overfit(n['box'], n2['box'])
                if s > 0.6:
                    used = True
            if used is False:
                dishes_choosed[6].append(n)

        elif n['class'] == 700:
            used = False
            for n2 in dishes_choosed[1]:
                s = overfit(n['box'], n2['box'])
                if s > 0.6:
                    used = True
            if used is False:
                dishes_choosed[6].append(n)
    # Экстра
    for n in dishes[0]:
        if n['class'] == 414 or n['class'] == 411:
            used = False
            for n2 in dishes_choosed[1]:
                s = overfit(n['box'], n2['box'])
                if s > 0.6:
                    dishes_choosed[1].remove(n2)
                    if 'add' not in n2:
                        n2['add'] = [n['class']]
                    else:
                        n2['add'].append(n['class'])
                    dishes_choosed[1].append(n2)
                        
            for n2 in dishes_choosed[2]:
                s = overfit(n['box'], n2['box'])
                if s > 0.6:
                    dishes_choosed[2].remove(n2)
                    if 'add' not in n2:
                        n2['add'] = [n['class']]
                    else:
                        n2['add'].append(n['class'])
                    dishes_choosed[2].append(n2)

    return dishes_choosed

font                   = cv2.FONT_HERSHEY_COMPLEX        
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2    

 
    

def plot_image(frame, 
               dishes, 
			   headers, 
			   ec):
    """
    

    Parameters
    ----------
    frame : cv2 image
        Изображение
    dishes : dict
        Распознанные блюда
    headers : list
        Загаловки групп блюд
    ec : list
        Цвета для отображения блюд

    Returns
    -------
    img : cv2 image
        Изображение со списком распознанных блюд
    sample : cv2 image
        Изображение с кадром и рамками распознанных блюд

    """
    it = 0
    ii = 0
    sample = copy.copy(frame)#.permute(1,2,0).cpu().numpy()
    img = np.zeros((1080,1000,3), np.uint8)
    for t in [1, 2, 3, 4, 5, 6, 7, 8]:
        bottomLeftCornerOfText = (100, 0 + 50 + 30*it)
        cv2.putText(img,headers[t],
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness = 1,
                    lineType = 2)
        it+=1
    
        for n in range(len(dishes[t])):
            
            start_point = (dishes[t][n]['box'][0], dishes[t][n]['box'][1])
            end_point = (dishes[t][n]['box'][2], dishes[t][n]['box'][3])
        
            thickness = 2
            sample = cv2.rectangle(sample, start_point, end_point, ec[ii], thickness)
                
            text = dishes[t][n]['text'] + ' = ' + str(np.round(dishes[t][n]['scores'], 2))
            for g in text.split('\n'):
                bottomLeftCornerOfText = (10, 0 + 50 + 30*it)
                cv2.putText(img, g,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            ec[ii],
                            thickness,
                            lineType)
        
                it += 1
                ii += 1
                
                if ii > 29:
                    ii = 0
                
        
    return img, sample
    
def test_image(dishes_choosed, 
               menu_day):
    """
    

    Parameters
    ----------
    dishes_choosed : dict
        Распознанные блюда по категориям
    menu_day : dict
        Дневное меню по категориям

    Returns
    -------
    dishes : dict
        Распознанные блюда по категориям с учетом дневного меню

    """
    dishes = {}

        
    for t in [1, 2, 3, 4, 5, 6, 7, 8]:
        
        dishes[t] = []
        
        
        a = menu_day[t].values[:, 2:7]
        ad = menu_day[t].values[:, 7:9]
        for n in dishes_choosed[t]:
            text = ''
                
            argw = np.argwhere(a == n['class'])
            if len(argw) == 0:
                if t == 5:
                    argw = np.argwhere(a == 803)
                if t == 2:
                    argw = np.argwhere(a == 101)
                        
            argw_add = []
            if 'add' in n:
                for added in n['add']:
                    argw_add.append(np.argwhere(ad == added))
                
                

            if len(argw_add) == 0:
                for ar in argw:
                    if len(text) == 0:
                        text = menu_day[t]['name'].values[ar[0]]
                    else:
                        text += ' # \n' + menu_day[t]['name'].values[ar[0]]
                    
                

            else:
                cls_ = {}
                for r in argw_add:
                    for k in r:
                        if k[0] not in cls_:
                            cls_[k[0]] = 1
                        else:
                            cls_[k[0]] += 1
                for r in argw:
                    if r[0] not in cls_:
                        cls_[r[0]] = 1
                    else:
                        cls_[r[0]] += 1
			    if len(cls_) > 0:
				    max_cl = max(cls_, key=cls_.get)
                    max_cl = cls_[max_cl]
                    for this in cls_:
                        if cls_[this] == max_cl:
                            if len(text) == 0:
                                text = menu_day[t]['name'].values[this]
                            else:
                                text += ' # \n' + menu_day[t]['name'].values[this]
             
            dish = {}
            dish['text'] = text
            dish['box'] = n['box']
            dish['scores'] = n['scores']
             
 
            dishes[t].append(dish)   
            
            
    return dishes 


def compare(dishes_old, 
            dishes_new):
    """
    

    Parameters
    ----------
    dishes_old : dict
        Набор распознанных блюд по категориям (предыдущее распознавание).
    dishes_new : dict
        Набор распознанных блюд по категориям (текущее распознавание).

    Returns
    -------
    dishes_concat : dict
        Набор распознанных блюд по категориям (объединение предыдущего и текущего распознавания).

    """
    dishes_concat = {}
    
    for t in [1, 2]:
        dishes_concat[t] = []
        for n in range(len(dishes_new[t])):
            dist = []
            ids = []
            for n_old in range(len(dishes_old[t])):
                if 'used' not in dishes_old[t][n_old]:
                    s = overfit(dishes_new[t][n]['box'], 
                                dishes_old[t][n_old]['box'])
                    dist.append(s)
                    ids.append(n_old)
             
            if len(dist) > 0:       
                df = pd.DataFrame()
                df['dist'] = dist
                df['id'] = ids
            
                a = np.argmax(df['dist'])
                n_old_ = df['id'].values[a]
            
                if np.max(df['dist']) > 0.4:
                    dishes_old[t][n_old_]['used'] = 1
                    dishes_new[t][n]['used'] = 1
                
                
                    if dishes_new[t][n]['scores'] > dishes_old[t][n_old_]['scores']:
                        if dishes_old[t][n_old_]['text'] in dishes_new[t][n]['text']:
                            dishes_new[t][n]['text'] = dishes_old[t][n_old_]['text']
                        dishes_concat[t].append(dishes_new[t][n])
                    
                    else:
                        if dishes_new[t][n]['text'] not in dishes_old[t][n_old_]['text']:
                            dishes_new[t][n]['text'] = dishes_old[t][n_old_]['text']
                        dishes_new[t][n]['scores'] = dishes_old[t][n_old_]['scores']
                        dishes_concat[t].append(dishes_new[t][n])
                        
                else:
                    dishes_new[t][n]['used'] = 1
                    dishes_concat[t].append(dishes_new[t][n])
            else:
                dishes_new[t][n]['used'] = 1
                dishes_concat[t].append(dishes_new[t][n])
                
    for t in [3, 4, 5, 6, 7, 8]:
        dishes_concat[t] = []
        for n in range(len(dishes_new[t])):
            dist = []
            ids = []
            if 'used' not in dishes_new[t][n]:
                
                for n_old in range(len(dishes_old[t])):
                    if dishes_new[t][n]['text'] == dishes_old[t][n_old]['text'] and 'used' not in dishes_old[t][n_old]:
                        s = overfit(dishes_new[t][n]['box'], 
                                    dishes_old[t][n_old]['box'])
                        dist.append(s)
                        ids.append(n_old)
                 
            if len(dist) > 0:       
                df = pd.DataFrame()
                df['dist'] = dist
                df['id'] = ids
                
                a = np.argmax(df['dist'])
                n_old_ = df['id'].values[a]
                
                dishes_old[t][n_old_]['used'] = 1
                dishes_new[t][n]['used'] = 1
                    
                if dishes_new[t][n]['scores'] > dishes_old[t][n_old_]['scores']:
                    if dishes_old[t][n_old_]['text'] in dishes_new[t][n]['text']:
                        dishes_new[t][n]['text'] = dishes_old[t][n_old_]['text']
                    dishes_concat[t].append(dishes_new[t][n])
                
                else:
                    if dishes_new[t][n]['text'] not in dishes_old[t][n_old_]['text']:
                        dishes_new[t][n]['text'] = dishes_old[t][n_old_]['text']
                    dishes_new[t][n]['scores'] = dishes_old[t][n_old_]['scores']
                    dishes_concat[t].append(dishes_new[t][n])                    
                   
    
    for t in [3, 4, 5, 6, 7, 8]:
        
       # """№
        for n in range(len(dishes_new[t])):
            dist = []
            ids = []
            for n_old in range(len(dishes_old[t])):
                if  'used' not in dishes_old[t][n_old]:
                    s = overfit(dishes_new[t][n]['box'], 
                                dishes_old[t][n_old]['box'])
                    dist.append(s)
                    ids.append(n_old)
                    
                    
            if len(dist) > 0:       
                df = pd.DataFrame()
                df['dist'] = dist
                df['id'] = ids
                
                a = np.argmax(df['dist'])
                n_old_ = df['id'].values[a]
                
                if np.max(df['dist']) > 0.9:
                    dishes_old[t][n_old_]['used'] = 1
                    dishes_new[t][n]['used'] = 1
                        
                        
                    if dishes_new[t][n]['scores'] > dishes_old[t][n_old_]['scores']:
                        if dishes_old[t][n_old_]['text'] in dishes_new[t][n]['text']:
                            dishes_new[t][n]['text'] = dishes_old[t][n_old_]['text']
                        dishes_concat[t].append(dishes_new[t][n])
                    
                    else:
                        if dishes_new[t][n]['text'] not in dishes_old[t][n_old_]['text']:
                            dishes_new[t][n]['text'] = dishes_old[t][n_old_]['text']
                        dishes_new[t][n]['scores'] = dishes_old[t][n_old_]['scores']
                        dishes_concat[t].append(dishes_new[t][n])
    
    

                            
        
    for t in [1 ,2, 3, 4, 5, 6, 7, 8]:
        for n_old in range(len(dishes_old[t])):
            if 'used' not in dishes_old[t][n_old]:
                dishes_old[t][n_old]['used'] = 1
                dishes_concat[t].append(dishes_old[t][n_old])
               
        for n_old in range(len(dishes_new[t])):
            if 'used' not in dishes_new[t][n_old]:
                dishes_new[t][n_old]['used'] = 1
                dishes_concat[t].append(dishes_new[t][n_old])
                
                
    for t in [1, 2, 3, 4, 5, 6, 7, 8]:
        for n in range(len(dishes_concat[t])):
            del dishes_concat[t][n]['used']
            
    drop = []        
    for t in [6]:
        for n in np.linspace(len(dishes_concat[t]) - 1, 0, len(dishes_concat[t])).astype(int):
            if dishes_concat[t][n]['text'] == 'Сметана 1/30':
                for t2 in [1]:
                    for n2 in range(len(dishes_concat[t2])): 
                        s = overfit(dishes_concat[t2][n2]['box'], 
                                    dishes_concat[t][n]['box'])
                        if s >= 0.9:
                            drop.append(n)
        for k in drop:
            del dishes_concat[t][k]
                
                    
    return dishes_concat
                    
 
def make_image(frame, dishes_, headers, ec, frame_width, frame_height):
    """
    Отображение результатов распознавания

    Parameters
    ----------
    frame : cv2 image
        изображение
    dishes_ : dict
        распознанные блюда по категориям
    headers : list
        заголовки подкатегорий
    ec : list
        список цветов
    frame_width : int
        высота фрейма
    frame_height : int
        ширина фрейма

    Returns
    -------
    hl : cv2 image
        отображение результатов распознавания

    """
    img, sample = plot_image(frame, dishes_, headers, ec)
    if img.shape[1] > 0 and sample.shape[1] > 0:
		hl = np.hstack((img, sample))
	else:
		raise ValueError("Empty image arrays")
    hl = cv2.resize(hl, (frame_width,frame_height), interpolation= cv2.INTER_LINEAR)
    return hl

def make_image_empty(frame, 
                     frame_width, 
					 frame_height):    
    """
    Отображение результатов распознавания (пустой поднос)

    Parameters
    ----------
    frame : cv2 image
        изображение
    frame_width : int
        высота фрейма
    frame_height : int
        ширина фрейма

    Returns
    -------
    hl : cv2 image
        отображение результатов распознавания (пустой поднос)

    """        
	
    img = np.zeros((1080,1000,3), np.uint8)
    if img.shape[1] > 0 and sample.shape[1] > 0:
		hl = np.hstack((img, sample))
	else:
		raise ValueError("Empty image arrays")
    hl = cv2.resize(hl, (frame_width,frame_height), interpolation= cv2.INTER_LINEAR) 
    
    return hl           
                
def make_menu(menu_filename):
    """
    Загрузка дневного меню и разбивка по категориям

    Parameters
    ----------
    menu_filename : text
        filename

    Returns
    -------
    menu_list : pd.DataFrame()
        полное дневное меню
    menu_day : dict
        дневное меню по категориям

    """
	
	if not os.path.exists(menu_filename):
		raise FileNotFoundError(f"File {menu_filename} not found")
    menu_list = pd.read_csv(menu_filename, sep = '\t')
    menu_day = {}
    for i in range(1, 9):
        menu_day[i] = menu_list[menu_list['type'] == i]
        
    return menu_list, menu_day



def make_classes_back(classes_filename: str) -> Dict:
    """
    Загрузка кодировки классов

    Parameters
    ----------
    classes_filename : text
        filename

    Returns
    -------
    classes_back : dict
        кодировка классов

    """
	if not os.path.exists(classes_filename):
		raise FileNotFoundError(f"File {classes_filename} not found")
    classes_back = joblib.load(classes_filename)
    return classes_back


def make_classes_values(classes_filename: str) -> Dict:
    """
    Загрузка значимости классов

    Parameters
    ----------
    classes_filename : text
        filename

    Returns
    -------
    class_values : dict
        значимость классов

    """
	if not os.path.exists(classes_filename):
		raise FileNotFoundError(f"File {classes_filename} not found")
    class_back = pd.read_csv(classes_filename)
    class_values = {}
    for i in range(class_back.shape[0]):
        class_values[class_back['id'].values[i]] = class_back['value'].values[i]
    return class_values



def load_model(model_filename, 
               classes_back):
    """
    Загрузка предобученной модели

    Parameters
    ----------
    model_filename : text
        filenane
    classes_back : dict
        кодировка классов
    Returns
    -------
    model : faster cnn resnet50
        предобученная модель
    """
    if not os.path.exists(model_filename):
		raise FileNotFoundError(f"File {model_filename} not found")
    if not os.path.exists(classes_back):
		raise FileNotFoundError(f"File {classes_back} not found")
		
    # load a model; pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = len(classes_back) + 1  # 1 class (wheat) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(model_filename
                               # , map_location=torch.device('cpu')
                                )
                     )
    
    return model              


def load_tea_model(model_name):
    """
    Загрузка модели распознавания чая

    Parameters
    ----------
    model_name : text
        filename

    Returns
    -------
    model : resnet18
        модель распознавания чая

    """
	if not os.path.exists(model_name):
		raise FileNotFoundError(f"File {model_name} not found")
		
    model = torchvision.models.resnet18()
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(*[
        nn.Linear(in_features=512, out_features=3),
        nn.Softmax(dim=1)
        ])

    a = torch.load(model_name, weights_only=False)
    model.load_state_dict(a)
    device = torch.device('cuda')
    model = model.to(device)
    return model

def test_tea(model, frame, device = torch.device('cuda')):
    """
    Распознавание чая

    Parameters
    ----------
    model : resnet18
        модель
    frame : cv2 image
        изображение
    device : cuda
        

    Returns
    -------
    bool
        True - обнаружен пакетик
        
        False - только кружка

    """
	
    f = np.array(frame)
    f = transform(frame).to(device)
    outputs = model(f.unsqueeze(0))
    outputs = outputs.cpu().detach().numpy()
    a = np.argmax(outputs)
	
    if a < 2:
        return True
    
	return False
