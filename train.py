import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
batchSize=2
imageSize=[600,600]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
trainDir="train"

imgs=[]
for pth in os.listdir(trainDir):
    imgs.append(trainDir+"/"+pth)
    
def loadData():
    batch_Imgs=[]
    batch_Data=[]# load images and masks
    for i in range(batchSize):
        idx=random.randint(0, len(imgs)-1)
        img = cv2.imread(imgs[idx])
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
        maskDir='train_masks'#'20220501_blue'
        # masks=[]
        # for mskName in os.listdir(maskDir):
        # if '(' in imgs[idx]:
        #    msk = imgs[idx].split('/')[-1].split('(')[0].strip()+'_blue(3).png'
        #    print(msk)
        # else:
        msk = imgs[idx].split('/')[-1].split('.')[0]+'_blue.png'

        print(maskDir+'\\'+msk)
        vesMask = (cv2.imread(maskDir+'\\'+msk, 0) > 0).astype(np.uint8)  # Read vesse instance mask
        vesMask=np.array(cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST))
            # masks.append(vesMask)# get bounding box coordinates for each mask
        num_objs = 1#len(masks)
        # if num_objs==0: return loadData() # if image have no objects just load another image
        boxes = torch.zeros([1,4], dtype=torch.float32)
        # for i in range(num_objs):
        x,y,w,h = cv2.boundingRect(vesMask[0])
        if w <= 0.0:
            continue
        print(i,y,w,h)
        boxes = torch.tensor([[x, y, x+w, y+h]])
        vesMask = torch.as_tensor(vesMask, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
        data["masks"] = vesMask
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
    if not batch_Imgs:
        return None, None
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=1)  # replace the pre-trained head with a new one
model.to(device)# move model to the right devic

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

for i in range(10001):
            images, targets = loadData()
            if images==None or targets==None:
                print('fail')
                continue
            images = list(image.to(device) for image in images)
            print(images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            print(i,'loss:', losses.item())
            # if i%10==0:
            torch.save(model.state_dict(), str(i)+".torch")