#!/usr/bin/env python3
# coding: utf-8

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random
import json
import gradio as gr
import numpy as np
import os
from io import BytesIO
from utilities import fetchOneImg
random.seed(0)
np.random.seed(0)




IMAGESROOTDIR = 'NINCO_OOD_classes'

class ImageDataset(Dataset):
    def __init__(self, annotation):
        self.annotation = pd.read_csv(annotation)
        self.batchFolder = 'imgBatch'
        if not os.path.exists(self.batchFolder):
            os.makedirs(self.batchFolder)

    def __getitem__(self, index):
        data_path = self.annotation.iloc[index,0]
        image = fetchOneImg(index, self.batchFolder)
        image = BytesIO(image)
        image = Image.open(image)
        label = self.annotation.iloc[index,1]
        source = 'imgBatch/'+data_path.split('/')[-1]
        return image, label, source

    def __len__(self):
        return len(self.annotation)

# instance of class ImageDataset
# contains all 765 images with their respective labels
imageDataset = ImageDataset(annotation='output.csv')

# Constants for the size of the images
SIZE = round(224/0.875)

# given an Index returns the transformed Image
# input: Index: int
# return: tuple(PIL Image, label)
def transform(index):
    assert index <= len(imageDataset)
    image, label, source = imageDataset[index]
    rescaledImage = fn.resize(img=image, size=[SIZE, SIZE], interpolation=T.InterpolationMode.BICUBIC)
    transformedImage = fn.center_crop(img=rescaledImage, output_size=[SIZE,SIZE])
    return transformedImage, label, source

# objects for tensor transformation
pilToTensor = T.ToTensor()
tensorToPil = T.ToPILImage()

# Class which is used to get the resized images with label
# input: datasetLength: int
# output:{'image': Tensor, 'label': String}
class DataLoader(Dataset):
    def __init__(self, datasetLength):
        self.datasetLength = datasetLength
   
    def __getitem__(self, index):
        self.index = index
        (picture, label, source) = transform(index)
        image = pilToTensor(picture)
        sample3dim = {'image' : image, 'label' : label}
        image = image.unsqueeze(0)
        sample = {'image': image, 'label': label}
        return sample, sample3dim, source


dataloader = DataLoader(len(imageDataset))

'''
function creates a random batch of data with a given size
Arguments: batchsize:int
Return: an array with a dict[image:label] 
'''
def createRandomBatch(batchsize, uId):
    # Number of tries to get another number
    random.seed(0)
    np.random.seed(0)
    TRIALSTHRESHOLD = 10000
    try:
        assert (0 < batchsize <= len(imageDataset))
    except AssertionError:
        errorFkt(f"Your batch size {batchsize} is not in the range 0 < batch size < Länge von {IMAGESROOTDIR} = {len(imageDataset)}")
    global attempts
    batch = []
    #batch3dim = []
    indexList = []
    sourceList = []
    labelList = []
    attempts = 0

    i = 0
    with open('data.json', 'r') as file:
        saves = json.load(file)
    while i < batchsize:
        i += 1
        if attempts >= TRIALSTHRESHOLD:
            errorFkt(f"The program tried more than {TRIALSTHRESHOLD} times to find an image which was not already shown to you. "
                     f"Please try to enter a smaller amount of tries than {len(indexList)}.")
        flag = False
        index = random.randint(0, len(imageDataset))
        if index in indexList:
            i -= 1
            attempts += 0.5
            flag = True

        for img in saves:
            if uId != None and img['ImgID'] == index:
                user_calls = img['UserCall']
                for call in user_calls:
                    if call['userId'] == int(uId):
                        i -= 1
                        attempts += 1
                        flag = True

        if flag:
            continue

        indexList.append(index)
        sample, sample3dim, source = dataloader[index]

        #batch3dim.append(sample3dim)
        sourceList.append(source)
        imgFile = source.split('/')[-1]
        batch.append(imgFile)
        label = sample['label']
        labelList.append(label)
    attempts = 0
    return batch, indexList, sourceList, labelList



'''
function extracts the values from the samples dict
Arguments: dict which contains random batch dict
Return: returns the values from samples list
'''
def extractValuesFromDict(samples, key:str):
    values = []
    for dictionary in samples:
        values.append(dictionary[key])
        
    if key == 'label':
        print(values)
    return values

# function to visualize the batch
def visualize(samples):
    tensors = extractValuesFromDict(samples, 'image')
    grid_border_size = 2
    elementsPerRow = 4
    grid = torchvision.utils.make_grid(tensor=tensors, nrow=elementsPerRow, padding=grid_border_size)
    plt.imshow(grid.detach().numpy().transpose((1,2,0)))





        
        



def errorFkt(text):

    with gr.Blocks() as demo:
        gr.Markdown(f'''{text}''')
        gr.Markdown('''Please restart the Program''')
    demo.launch()

