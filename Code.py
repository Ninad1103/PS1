import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # utils
# import torch.optim as optim


import cv2
import matplotlib.pyplot as plt
import imutils

import numpy as np
from PIL import Image
import glob

from numpy import asarray

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

#----------------------------------------------------------------------------------------------------------

def transform(pos):
# This function is used to find the corners of the object and the dimensions of the object
    pts=[]
    n=len(pos)
    for i in range(n):
        pts.append(list(pos[i][0]))

    sums={}
    diffs={}
    tl=tr=bl=br=0
    for i in pts:
        x=i[0]
        y=i[1]
        sum=x+y
        diff=y-x
        sums[sum]=i
        diffs[diff]=i
    sums=sorted(sums.items())
    diffs=sorted(diffs.items())
    n=len(sums)
    rect=[sums[0][1],diffs[0][1],diffs[n-1][1],sums[n-1][1]]
    #      top-left   top-right   bottom-left   bottom-right

    h1=np.sqrt((rect[0][0]-rect[2][0])**2 + (rect[0][1]-rect[2][1])**2)     #height of left side
    h2=np.sqrt((rect[1][0]-rect[3][0])**2 + (rect[1][1]-rect[3][1])**2)     #height of right side
    h=max(h1,h2)

    w1=np.sqrt((rect[0][0]-rect[1][0])**2 + (rect[0][1]-rect[1][1])**2)     #width of upper side
    w2=np.sqrt((rect[2][0]-rect[3][0])**2 + (rect[2][1]-rect[3][1])**2)     #width of lower side
    w=max(w1,w2)

    return int(w),int(h),rect

def save_final_results(noI):
    image_dir_masked = './test_data/u2netp_results/'
    imo_name = cv2.imread(image_dir_masked+noI[0]+'.png',0)
    image_dir = './test_data/test_images/'
    org_image=cv2.imread(image_dir+noI[0]+'.'+noI[1])
    #-------------------------------------------------------------------------------------
    edges1 = cv2.Canny(org_image,150,200,3)

    img = cv2.cvtColor(imo_name,cv2.COLOR_BGR2RGB)
    vectorized = img.reshape((-1,3))

    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10 , 1.0)

    k=3
    attempts=10
    ret, label, center = cv2.kmeans(vectorized,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    edges = cv2.Canny(result_image,150,200,3)
    #---------------------------------------------------------------------------------------
#finding all the contours
    allContours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#list of contours in image , each contour is a numpy array(x,y) cordinates of the boundary points of the object
    allContours = imutils.grab_contours(allContours)
    print(len(allContours))

#sortind the contours on the basis of their area in descending order
    allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]
#approximating the contour to a polygon

    perimeter = cv2.arcLength(allContours[0], True)
    ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)

    img1 = cv2.drawContours(org_image, [ROIdimensions], -1, (0,255,0), 2)
# -1--> draws all contours, (0,255,0) --> tells colors,2-->thickness

    plt.imshow(img1,cmap = 'gray')
    plt.show()
    #-------------------------------------------------------------------------------------------------------
    w,h,arr=transform(ROIdimensions)

    pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts1=np.float32(arr)
    M=cv2.getPerspectiveTransform(pts1,pts2)
    dst=cv2.warpPerspective(org_image,M,(w,h))
    #image_f=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    #image_f=cv2.adaptiveThreshold(image_f,255,1,0,11,2)
    imag2 = cv2.resize(dst,(w,h),interpolation = cv2.INTER_AREA)
    #imag2 = cv2.cvtColor(imag2,cv2.COLOR_GRAY2RGB)
    plt.imshow(imag2,cmap = 'gray')
    plt.show()



#--------------------------------------------------------------------------------------------------------
def main():

    # --------- 1. get image path and name ---------
    model_name='u2netp'#change to u2netp


    image_dir = './test_data/test_images/'
    prediction_dir = './test_data/' + model_name + '_results/'
    model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'
    #----------- input name and format of image -------
    NOI=input("Enter name of image (with it's format):  ")
    noI = NOI.rsplit('.', 1)


    img_name_list = glob.glob(image_dir + NOI)
    org_img=img_name_list
    type(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))## this line is added map_location='cpu'
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test],pred,prediction_dir)
        save_final_results(noI)#new code

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
