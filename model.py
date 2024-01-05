import os
import cv2
import network, utils
import torch
import torch.nn as nn
import albumentations as A
import albumentations.pytorch
from PIL import Image
from custom import CustomSegmentation
import numpy as np
from torch.autograd import Variable
from DCSAU_Net import DCSAU_Net

def yuv_color_format(load_img):
    image_yuv = cv2.cvtColor(load_img, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    return image_rgb
    
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

def resizing(crop_img):
    x_p,y_p = 512,512
    percent = 1
    if(crop_img.shape[1] > crop_img.shape[0]) :
        percent = x_p/crop_img.shape[1]
    else :
        percent = y_p/crop_img.shape[0]
    crop_img = cv2.resize(crop_img, dsize=(0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_NEAREST)
    y,x,h,w = (0,0,crop_img.shape[0], crop_img.shape[1])
    w_x = (x_p-(w-x))/2 
    h_y = (y_p-(h-y))/2
    if(w_x < 0):
        w_x = 0
    elif(h_y < 0):
        h_y = 0
    M = np.float32([[1,0,w_x], [0,1,h_y]])
    img_re = cv2.warpAffine(crop_img, M, (x_p, y_p))
    
    return img_re

def image_preprocessing(load_img):
    # resized_image = cv2.resize(load_img, (4288, 2848))
    # crop_img = resized_image[920:-1210, 820:-820]
    # right = crop_img[:, :crop_img.shape[1]//2]
    # right = right[200:, 450:]
    # left = crop_img[:, crop_img.shape[1]//2:]
    # left = left[200:, :-450]
    
    # right = resizing(right)
    # left = resizing(left)
    
    # half = crop_img.shape[1]//2
    # sstt = crop_img[:, half-half//2:half+half//2]
    # sstt = sstt[:-200, 250:-250]
    # sticker = resizing(sstt)
    
    right = load_img[:, :load_img.shape[1]//2]
    right = right[200:, 450:]
    left = load_img[:, load_img.shape[1]//2:]
    left = left[200:, :-450]

    right = resizing(right)
    left = resizing(left)
    
    half = load_img.shape[1]//2
    sstt = load_img[:, half-half//2:half+half//2]
    sstt = sstt[:-200, 250:-250]
    sticker = resizing(sstt)
    
    return load_img, left, right, sticker

def seg_predict(d_path, f_name):
    origin_path, l_path, r_path, s_path, ov_path = dir_check()
    load_img = cv2.imread(d_path+'/'+f_name)
    original, left, right, sticker = image_preprocessing(load_img)
    
    f_name = f_name.replace('.JPG', '').replace('.jpg', '').replace('.PNG', '').replace('.png', '')
    cv2.imwrite(origin_path+'/'+f_name+'.png', original)
    cv2.imwrite(l_path+'/'+f_name+'_L.png', left)   # ./cut/cut_left/f_name_L.png
    cv2.imwrite(r_path+'/'+f_name+'_R.png', right)  # ./cut/cut_right/f_name_R.png
    cv2.imwrite(s_path+'/'+f_name+'_s.png', sticker)    # ./cut/cut_sticker/f_name_s.png
    
    # left = yuv_color_format(left)
    # right = yuv_color_format(right)
    sticker = yuv_color_format(sticker)
    
    f_name = f_name.replace('.JPG', '').replace('.jpg', '').replace('.PNG', '').replace('.png', '')
    
    decode_fn = CustomSegmentation.decode_target
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            A.pytorch.transforms.ToTensorV2()   
        ])

    L_img = transform(image=left)['image'].unsqueeze(0)
    R_img = transform(image=right)['image'].unsqueeze(0)
    S_img = transform(image=sticker)['image'].unsqueeze(0)
    L_img = L_img.to(device)
    R_img = R_img.to(device)
    S_img = S_img.to(device)
 
    model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=2, output_stride=16)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # sclera
    model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=2, output_stride=16)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    s_ckpt = torch.load('./path_file/sclera-best_deeplabv3plus_resnet101_custom_os16.pth', map_location=torch.device('cpu'))
    model.load_state_dict(s_ckpt["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    with torch.no_grad():
        s_model = model.eval()
        # Left
        L_s_pred = s_model(L_img).max(1)[1].cpu().numpy()[0]
        colorized_L_s = decode_fn(L_s_pred).astype('uint8')
        colorized_L_s = Image.fromarray(colorized_L_s)
        colorized_L_s.save(os.path.join(l_path+'/', f_name+'_L_sclera_pred'+'.png'))
        # Right
        R_s_pred = s_model(R_img).max(1)[1].cpu().numpy()[0] 
        colorized_R_s = decode_fn(R_s_pred).astype('uint8')
        colorized_R_s = Image.fromarray(colorized_R_s)
        colorized_R_s.save(os.path.join(r_path+'/', f_name+'_R_sclera_pred'+'.png'))
    print(' - Scelra predict')
    
    # iris
    model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=2, output_stride=16)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    i_ckpt = torch.load('./path_file/iris-best_deeplabv3plus_resnet101_custom_os16.pth', map_location=torch.device('cpu'))
    model.load_state_dict(i_ckpt["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    with torch.no_grad():
        i_model = model.eval()
        # Left
        L_i_pred = i_model(L_img).max(1)[1].cpu().numpy()[0] # HW
        colorized_L_i = decode_fn(L_i_pred).astype('uint8')
        colorized_L_i = Image.fromarray(colorized_L_i)
        colorized_L_i.save(os.path.join(l_path+'/', f_name+'_L_iris_pred'+'.png'))
        # Right
        R_i_pred = i_model(R_img).max(1)[1].cpu().numpy()[0] # HW
        colorized_R_i = decode_fn(R_i_pred).astype('uint8')
        colorized_R_i = Image.fromarray(colorized_R_i)
        colorized_R_i.save(os.path.join(r_path+'/', f_name+'_R_iris_pred'+'.png'))
    print(' - Iris predict')
    
    # reflex - deeplab
    model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=2, output_stride=16)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    # i_ckpt = torch.load('./path_file/reflex-(focal)-best_deeplabv3plus_resnet101_custom_os16.pth', map_location=torch.device('cpu'))
    i_ckpt = torch.load('./path_file/ori_size.pth', map_location=torch.device('cpu'))
    # i_ckpt = torch.load('./path_file/crop_size.pth', map_location=torch.device('cpu'))
    model.load_state_dict(i_ckpt["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    
    with torch.no_grad():
        r_model = model.eval()
        # Left
        L_r_pred = r_model(L_img).max(1)[1].cpu().numpy()[0] # HW
        colorized_L_r = decode_fn(L_r_pred).astype('uint8')
        colorized_L_r = Image.fromarray(colorized_L_r)
        colorized_L_r.save(os.path.join(l_path+'/', f_name+'_L_reflex_pred'+'.png'))
        # Right
        R_r_pred = r_model(R_img).max(1)[1].cpu().numpy()[0] # HW
        colorized_R_r = decode_fn(R_r_pred).astype('uint8')
        colorized_R_r = Image.fromarray(colorized_R_r)
        colorized_R_r.save(os.path.join(r_path+'/', f_name+'_R_reflex_pred'+'.png'))
   
   
    # reflex - dcsau-net    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # model = torch.load('./path_file/reflex.pth', map_location=torch.device('cpu'))
    
    # model_path = './path_file/reflex.pth'
    # model = DCSAU_Net()
    # model.load_state_dict(torch.load(model_path, map_location=device))
    
    # model = nn.DataParallel(model)
    # model.to(device)
    # r_L_img = transform(image=left)['image']
    # r_R_img = transform(image=right)['image']
    # r_L_img = Variable(torch.unsqueeze(r_L_img, dim=0).float(), requires_grad=False)
    # r_R_img = Variable(torch.unsqueeze(r_R_img, dim=0).float(), requires_grad=False)
    # with torch.no_grad():
    #     r_model = model.eval()
    #     # Left
    #     L_r_pred = r_model(r_L_img)
    #     L_r_pred = torch.sigmoid(L_r_pred)
    #     L_r_pred[L_r_pred >= 0.5] = 1
    #     L_r_pred[L_r_pred < 0.5] = 0
    #     L_r_pred_draw = L_r_pred.clone().detach()
    #     L_r_numpy = L_r_pred_draw.cpu().detach().numpy()[0][0]
    #     L_r_numpy[L_r_numpy==1] = 255
    #     cv2.imwrite(l_path+'/'+f_name+'_L_reflex_pred'+'.png', L_r_numpy)
    #     R_r_pred = r_model(r_R_img)
    #     R_r_pred = torch.sigmoid(R_r_pred)
    #     R_r_pred[R_r_pred >= 0.5] = 1
    #     R_r_pred[R_r_pred < 0.5] = 0
    #     R_r_pred_draw = R_r_pred.clone().detach()
    #     R_r_numpy = R_r_pred_draw.cpu().detach().numpy()[0][0]
    #     R_r_numpy[R_r_numpy==1] = 255
    #     cv2.imwrite(r_path+'/'+f_name+'_R_reflex_pred'+'.png', R_r_numpy)
    print(' - Reflex predict')
    
    # sticker
    model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=2, output_stride=16)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    st_ckpt = torch.load('./path_file/sticker-best_deeplabv3plus_resnet101_custom_os16.pth', map_location=torch.device('cpu'))
    model.load_state_dict(st_ckpt["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    with torch.no_grad():
        st_model = model.eval()
        st_pred = st_model(S_img).max(1)[1].cpu().numpy()[0] # HW
        colorized_st = decode_fn(st_pred).astype('uint8')
        colorized_st = Image.fromarray(colorized_st)
        colorized_st.save(os.path.join(s_path+'/', f_name+'_s_sticker_pred'+'.png'))
    print(' - Sticker predict')
    
if __name__ == '__main__':
    destination_path = "./original/"
    file_name = 'test1.jpg'
    seg_predict(destination_path, file_name)
    