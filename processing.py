import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image

class eye:
    # left/right end point 
    def end_point(gray_arr):
        left_point, right_point = [], []
        for i in range(gray_arr.shape[0]):
            for j in range(gray_arr.shape[1]):
                try:
                    if gray_arr[i][j] != 0 and gray_arr[i][j-1] == 0: 
                        left_point.append([i, j])
                    elif gray_arr[i][j] != 0 and gray_arr[i][j+1] == 0:
                        right_point.append([i, j])
                except:
                    if gray_arr[i][j] != 0 and j == 0: 
                        left_point.append([i, j])
                    elif gray_arr[i][j] != 0 and j == gray_arr.shape[1]-1:
                        right_point.append([i, j])
        if len(left_point) == 0:
            left_point = [[0,0]]
        if len(right_point) == 0:
            right_point = [gray_arr.shape[0],gray_arr.shape[1]]
            
        left_point, right_point = np.array(left_point).T, np.array(right_point).T
        left_point[0], left_point[1] = np.flip(left_point[0]), np.flip(left_point[1])
        right_point[0], right_point[1] = np.flip(right_point[0]), np.flip(right_point[1])
        min_left_index, max_right_index = np.argmin(left_point[1]), np.argmax(right_point[1])
        min_left = (left_point[0][min_left_index], left_point[1][min_left_index])
        max_right = (right_point[0][max_right_index], right_point[1][max_right_index])

        return min_left, max_right
    
    # reflex center coordinate
    def center_dot(img_arr):
        center_set = []
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                if img_arr[i][j] != 0:
                    center_set.append([i, j])
        center_set = np.array(center_set).T
        center = [round(np.mean(center_set[0])), round(np.mean(center_set[1]))]
        
        return center
    
class sticker:
    # measure diameter
    def diameter(img_arr):
        horizontal_diameter = []
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                # horizontal
                if img_arr[i][j] != 0 and img_arr[i][j-1] == 0:
                    for k in range(j, img_arr.shape[1]-1):
                        if img_arr[i][k] != 0 and img_arr[i][k+1] == 0:
                            horizontal_diameter.append(k-j)
                
        if len(horizontal_diameter) != 0:
            return max(horizontal_diameter)
        else:
            return 0

class coloring:   
    # coloring Iris
    def colored_iris(img_arr, iris):
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                if all(iris[i][j] != [0, 0, 0]):
                    img_arr[i][j] = [0, 64, 0]   
        return img_arr
    
    # coloring edge
    def colored_edge(img_arr, upper_edge, lower_edge):
        for eg in lower_edge:
            img_arr[eg[0]][eg[1]] = [0, 255, 255]
        for eg in upper_edge:
            img_arr[eg[0]][eg[1]] = [0, 255, 0]
        
        return img_arr
    
    # coloring MRD1, MRD2
    def colored_PF(img_arr, gray_img, center):
        for i in range(0, gray_img.shape[0]):
            if all(gray_img[i][center[1]] != [0, 0, 0]):
                if i < center[0]-1:
                    img_arr[i][center[1]] = [255, 0, 0]
                elif i > center[0]+1:
                    img_arr[i][center[1]] = [0, 128, 255]
        return img_arr
    
    
    # coloring center_dot(reflex)
    def colored_center(img_arr, center):
        i, j =  center[0], center[1]
        img_arr[i-1][j-1] = [255, 255, 255]
        img_arr[i-1][j] = [255, 255, 255]
        img_arr[i-1][j+1] = [255, 255, 255]
        img_arr[i][j-1] = [255, 255, 255]
        img_arr[i][j] = [255, 255, 255]
        img_arr[i][j+1] = [255, 255, 255]
        img_arr[i+1][j-1] = [255, 255, 255]
        img_arr[i+1][j] = [255, 255, 255]
        img_arr[i+1][j+1] = [255, 255, 255]
        
        return img_arr