import cv2, math
import numpy as np
from PIL import Image

def place_y(center, edge):
    y_dot_1, y_dot_2 = center, center
    while True:
        y_dot_1 += 1
        y_dot_2 -= 1
        if y_dot_1 in edge:
            y_dot = y_dot_1
            break
        elif y_dot_2 in edge:
            y_dot = y_dot_2
            break
        if y_dot_1 == 512 or y_dot_2 == 0: break
    return y_dot

class pixel_length:
    # mesure curved length
    def curve_length(img_arr, min_left, max_right):
        upper_lid_edge, lower_lid_edge = [], []
        center_j = (max_right[1]+min_left[1])//2
        upper_lid_length, lower_lid_length = 0, 0
        
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                if img_arr[i][j] != 0:
                    # left - lower
                    if j <= center_j and i >= min_left[0]:
                        # edge 
                        if img_arr[i][j-1] == 0 or img_arr[i+1][j] == 0:
                            lower_lid_edge.append([i, j])
                        if img_arr[i][j-1] == 0 or img_arr[i+1][j] == 0:
                            lower_lid_edge.append([i, j])
                        # length
                        if img_arr[i][j-1] == 0 and img_arr[i+1][j] == 0:
                            if img_arr[i][j+1] == 0:    lower_lid_length += 4.14
                            else:    lower_lid_length += 1.57
                        elif img_arr[i][j-1] != 0 and img_arr[i][j+1] != 0 and img_arr[i+1][j] == 0:    
                            lower_lid_length += 1
                    # left - upper
                    elif j <= center_j and i < min_left[0]:
                        # edge 
                        if img_arr[i][j-1] == 0 or img_arr[i-1][j] == 0:
                            upper_lid_edge.append([i, j])
                        # length
                        if img_arr[i][j-1] == 0 and img_arr[i-1][j] == 0:
                            if img_arr[i][j+1] == 0:    upper_lid_length += 4.14
                            else:    upper_lid_length += 1.57
                        elif img_arr[i][j-1] != 0 and img_arr[i][j+1] != 0 and img_arr[i-1][j] == 0:    
                            upper_lid_length += 1
                    
                    # right - lower
                    elif j > center_j and i >= max_right[0]:
                        # edge
                        if img_arr[i][j+1] == 0 or img_arr[i+1][j] == 0:
                            lower_lid_edge.append([i, j])
                        # length
                        if img_arr[i][j+1] == 0 and img_arr[i+1][j] == 0:
                            if img_arr[i][j-1] == 0:    lower_lid_length += 4.14
                            else:    lower_lid_length += 1.57
                        elif img_arr[i][j-1] != 0 and img_arr[i][j+1] != 0 and img_arr[i+1][j] == 0:    
                            lower_lid_length += 1       
                            
                    # right - upper
                    elif j > center_j and i < max_right[0]:
                        # edge
                        if img_arr[i][j+1] == 0 or img_arr[i-1][j] == 0:
                            upper_lid_edge.append([i, j])
                        # length
                        if img_arr[i][j+1] == 0 and img_arr[i-1][j] == 0:
                            if img_arr[i][j-1] == 0:    upper_lid_length += 4.14
                            else:    upper_lid_length += 1.57
                        elif img_arr[i][j-1] != 0 and img_arr[i+1][j] != 0 and img_arr[i-1][j] == 0:    
                            upper_lid_length += 1

        return upper_lid_length, lower_lid_length, upper_lid_edge, lower_lid_edge
    
    # mesure straight length
    def straight_length(center, upper_edge, lower_edge):
        upper_edge, lower_edge = np.array(upper_edge).T, np.array(lower_edge).T
        try:
            MRD1 = center[0]-upper_edge[0][list(upper_edge[1]).index(center[1])]
            MRD2 = lower_edge[0][list(lower_edge[1]).index(center[1])] - center[0]
        except:
            y_dot = place_y(center[1], upper_edge)
            MRD1 = center[0]-upper_edge[0][list(upper_edge[1]).index(y_dot)]
            y_dot = place_y(center[1], lower_edge)
            MRD2 = lower_edge[0][list(lower_edge[1]).index(y_dot)] - center[0]

        return MRD1, MRD2


class real_length:
    # mesure real length
    def horizontal_length(pixel_list, diameter):
        # sticker diameter: 16mm
        st_d = 9
        
        # Must be colored using colored_edge in processing
        upper_lid, lower_lid = pixel_list[0], pixel_list[1]
        mrd1, mrd2 = pixel_list[2], pixel_list[3]
        ho_di = diameter
        
        h_real_upper = round(st_d*upper_lid / ho_di, 2)
        h_real_lower = round(st_d*lower_lid / ho_di, 2)
        h_MRD1 = round(st_d*mrd1 / ho_di, 2)
        h_MRD2 = round(st_d*mrd2 / ho_di, 2)
        
        return [h_real_upper, h_real_lower, h_MRD1, h_MRD2]

def medial_canthus(left, right):
    a = abs(left[0] - right[0])
    b = abs(left[1] - right[1])
    c = math.sqrt(a**2 + b**2)
    x = math.acos(a/c)
    return round(x, 4)