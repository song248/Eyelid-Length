import os
import cv2
import processing, measure
import pandas as pd
import numpy as np
import model
from datetime import datetime
from tqdm import tqdm

def dir_check():
    
    cut_dir = './cut'
    origin_path = "./cut/cut_origin"
    l_path = "./cut/cut_left"
    r_path = "./cut/cut_right"
    s_path = "./cut/cut_sticker"
    result_path = './result'
    ov_result_path = './ov_result'
    
    if not os.path.exists(cut_dir):
        os.makedirs(cut_dir)
    if not os.path.exists(origin_path):
        os.makedirs(origin_path)
    if not os.path.exists(l_path):
        os.makedirs(l_path)
    if not os.path.exists(r_path):
        os.makedirs(r_path)
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path) 
    if not os.path.exists(ov_result_path):
        os.makedirs(ov_result_path) 
    
    return origin_path, l_path, r_path, s_path, ov_result_path


def length_measure(f_name, path, s_path, L_R):
    # diameter
    sticker = cv2.imread(s_path+'/'+f_name+'_s_sticker_pred.png')
    sticker = cv2.cvtColor(sticker, cv2.COLOR_BGR2GRAY)
    horizontal_diameter = processing.sticker.diameter(sticker)
        
    if L_R =='L':
        f_name = f_name+'_L'
        print(' - L:', end=" ")
    else: 
        f_name = f_name+'_R'
        print(' - R:', end=" ")
    
    load_img = cv2.imread(path+'/'+f_name+'_sclera_pred.png')
    gray_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2GRAY)
    load_reflex = cv2.imread(path+'/'+f_name+'_reflex_pred.png')
    gray_reflex = cv2.cvtColor(load_reflex, cv2.COLOR_BGR2GRAY)
    load_iris = cv2.imread(path+'/'+f_name+'_iris_pred.png')
    gray_iris = cv2.cvtColor(load_iris, cv2.COLOR_BGR2GRAY)
    
    # try:
    try:
        center = processing.eye.center_dot(gray_reflex) 
    except:
        try:
            center = processing.eye.center_dot(gray_iris)
        except:
            center = processing.eye.center_dot(gray_img)

    # end point
    min_left, max_right = processing.eye.end_point(gray_img)
    # length
    upper_lid_length, lower_lid_length, upper_edge, lower_edge = measure.pixel_length.curve_length(gray_img, min_left, max_right)
    mrd1, mrd2 = measure.pixel_length.straight_length(center, upper_edge, lower_edge)

    # measure
    pixel_length = [upper_lid_length, lower_lid_length, mrd1, mrd2]
    horizon = measure.real_length.horizontal_length(pixel_length, horizontal_diameter)
    print(horizon)

    # result save
    save_img = processing.coloring.colored_iris(load_img, gray_iris)
    save_img = processing.coloring.colored_center(save_img, center)
    save_img = processing.coloring.colored_PF(save_img, gray_img, center)
    save_img = processing.coloring.colored_edge(save_img, upper_edge, lower_edge)
    
    cv2.imwrite('./result/'+f_name+'_result.png', save_img)

    return horizon, save_img[50:-50, 50:-50], [upper_edge, lower_edge], horizontal_diameter
    

def input_image_handle(d_path, f_name):
    print('< model predict>')
    origin_path, l_path, r_path, s_path, ov_path = dir_check()
    model.seg_predict(d_path, f_name)
    f_name = f_name.replace('.JPG', '').replace('.jpg', '').replace('.PNG', '').replace('.png', '')
    
    print('')
    print('< Length Measuremnet >')
    # upper, lower, mrd1, mrd2
    L_length, L_img, L_edge, diam = length_measure(f_name, l_path, s_path, 'L')
    R_length, R_img, R_edge, diam = length_measure(f_name, r_path, s_path, 'R')
    
    # make overlay image
    origin_L = cv2.imread(l_path+'/'+f_name+'_L.png')
    origin_R = cv2.imread(r_path+'/'+f_name+'_R.png')
    o_t_L = cv2.imread('./result/'+f_name+'_L_result.png')
    o_t_R = cv2.imread('./result/'+f_name+'_R_result.png')
    overlay_L = cv2.addWeighted(origin_L, 0.5, o_t_L, 0.5, 0)
    overlay_R = cv2.addWeighted(origin_R, 0.5, o_t_R, 0.5, 0)
    
    origin_L = cv2.cvtColor(origin_L, cv2.COLOR_BGR2RGB)
    origin_R = cv2.cvtColor(origin_R, cv2.COLOR_BGR2RGB)
    L_img = cv2.cvtColor(L_img, cv2.COLOR_BGR2RGB)
    R_img = cv2.cvtColor(R_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(ov_path+'/'+f_name+'_L_overlay.png', overlay_L)
    cv2.imwrite(ov_path+'/'+f_name+'_R_overlay.png', overlay_R)
    overlay_L = cv2.cvtColor(overlay_L, cv2.COLOR_BGR2RGB)
    overlay_R = cv2.cvtColor(overlay_R, cv2.COLOR_BGR2RGB)
    
    return origin_L, origin_R,\
            L_length, R_length,\
            L_img, R_img,\
            overlay_L, overlay_R,\
            L_edge, R_edge, f_name, diam

def load_result(f_name, edge, diam, s):
    origin_path, l_path, r_path, s_path, ov_path = dir_check()
    if s == 'L':
        load_img = cv2.imread(ov_path+'/'+f_name+'_L_overlay.png')
        load_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB)
    if s == 'R':
        load_img = cv2.imread(ov_path+'/'+f_name+'_R_overlay.png')
        load_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB)
    edge_one = sorted(edge[0], key=lambda x: (x[1], -x[0]))
    edge_two = sorted(edge[1], key=lambda x: (x[1], -x[0]))
    return load_img, [edge_one, edge_two], diam

def many_to_save(f_list):
    df = pd.DataFrame(columns=['file_name', 'L_Upper_Lid', 'L_Lower_Lid', 'L_MRD1', 'L_MRD2', 'R_Upper_Lid', 'R_Lower_Lid', 'R_MRD1', 'R_MRD2'])
    for i in tqdm(range(len(f_list))):
        path = f_list[i]
        d_path = os.path.dirname(path)
        f_name = os.path.basename(path)
        o_l, o_r, L_length, R_length, L_img, R_img,\
                overlay_L, overlay_R, L_edge, R_edge, f_name, diam = input_image_handle(d_path, f_name)
        df = df.append({'file_name': f_name,\
                        'L_Upper_Lid': L_length[0], 'L_Lower_Lid': L_length[1], 'L_MRD1': L_length[2], 'L_MRD2': L_length[3],
                        'R_Upper_Lid': R_length[0], 'R_Lower_Lid': R_length[1], 'R_MRD1': R_length[2], 'R_MRD2': R_length[3]}, ignore_index=True)
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    excel_filename = f'data_{date_time}.xlsx'
    df.to_excel(excel_filename, index=False)
    

        