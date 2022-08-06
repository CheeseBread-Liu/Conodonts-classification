# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:11:13 2020

@author: Liu Tianzi
"""

import torch
import torch.nn as nn
from model import resnet34, resnet18, resnet50, resnet101, resnet152
from DenseNetmodel import densenet121
from EfficientNetmodel import efficientnet_b2,efficientnet_b0,efficientnet_b1,efficientnet_b3
from RegNetmodel import create_regnet
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os
import xlwt

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = './all/'

# create model
model = resnet152(num_classes=8)

model = nn.DataParallel(model)
model.to(device)
# load model weights
model_weight_path = "./resNet152-pre.pth"
model.load_state_dict(torch.load(model_weight_path))


# create Excel
count=0
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("ResNet152")

sheet.write(count,0, '序号') # row, column, value
sheet.write(count,1, '种类')
sheet.write(count,2, '拍摄方式')
sheet.write(count,3, '识别结果')
sheet.write(count,4, '是否识别正确')
sheet.write(count,5, '预测准确率')
count=count+1

ImgID = []
ImgClass = []
ImgShootingMode = []
ImgIdentResult = []
Correct = []
PredictProbability = []

for path in os.listdir(root_path):
    
    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
        json_file = open('./shooting_mode.json', 'r')
        shooting_mode = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)
    
    ImgID.append(path.split('.')[0])
    ImgClass.append(class_indict[path.split('.')[0][0]].split(' ')[0])
    ImgShootingMode.append(shooting_mode[path.split('.')[0][1]])
    # load image
    img = Image.open(root_path + path)
    #plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device)))
        # output = model(img.to(device))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).cpu().numpy()
    ImgIdentResult.append(class_indict[str(predict_cla)].split(' ')[0])
    PredictProbability.append(predict[predict_cla].cpu().numpy())
    if class_indict[str(predict_cla)].split(' ')[0] == class_indict[path.split('.')[0][0]].split(' ')[0]:
        Correct.append('正确')
    else:
        Correct.append('错误')
    print(class_indict[str(predict_cla)], predict[predict_cla].cpu().numpy())
    
# write Excel
for i in range(len(os.listdir(root_path))):
    sheet.write(count,0, str(ImgID[i])) # row, column, value
    sheet.write(count,1, str(ImgClass[i]))
    sheet.write(count,2, str(ImgShootingMode[i]))
    sheet.write(count,3, str(ImgIdentResult[i]))
    sheet.write(count,4, str(Correct[i]))
    sheet.write(count,5, str(PredictProbability[i]))
    count=count+1
workbook.save('ResNet152.xls')

