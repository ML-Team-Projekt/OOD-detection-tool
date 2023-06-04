#!/usr/bin/env python3
# coding: utf-8

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from  utilities import createAnnotation
from model_loader import get_new_model
import pandas as pd
from PIL import Image
import random
import class_katalog
import json
import gradio as gr


IMAGESROOTDIR = 'NINCO_OOD_classes'

class ImageDataset(Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir
        createAnnotation(self.rootDir)
        self.annotation = pd.read_csv('output.csv')


    def __getitem__(self, index):
        data_path = self.annotation.iloc[index,0]
        image = Image.open(data_path)
        label = self.annotation.iloc[index,1]
        source = data_path
        return image, label, source

    def __len__(self):
        return len(self.annotation)

# instance of class ImageDataset
# contains all 765 images with their respective labels
imageDataset = ImageDataset(rootDir=IMAGESROOTDIR)

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

#dataloader = DataLoader(len(imageDataset))
#iterr = iter(dataloader)
#im, lab, source = next(iterr)
#print(im['image'].size())

# Amount of random samples 
#BATCHSIZE = 4

dataloader = DataLoader(len(imageDataset))



'''
function creates a random batch of data with a given size
Arguments: batchsize:int
Return: an array with a dict[image:label] 
'''
def createRandomBatch(batchsize, uId):
    # Number of tries to get another Number
    TRIALSTHRESHOLD = 100
    try:
        assert (0 < batchsize <= len(imageDataset))
    except AssertionError:
        errorFkt(f"Your batch size {batchsize} is not in the range 0 < batch size < Länge von {IMAGESROOTDIR} = {len(imageDataset)}")
    global attempts
    batch = []
    batch3dim = []
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
            if img['ImgID'] == index:
                user_calls = img['UserCall']
                for call in user_calls:
                    if call['userId'] == int(uId):
                        i -= 1
                        attempts += 1
                        #print(attempts)
                        flag = True

        if flag:
            continue

        indexList.append(index)
        sample, sample3dim, source = dataloader[index]
        batch.append(sample)
        batch3dim.append(sample3dim)
        sourceList.append(source)
        label = sample['label']
        labelList.append(label)
    attempts = 0
    return batch, batch3dim, indexList, sourceList, labelList

#samples, samples3dim, indexList, sourceList, labelList = createRandomBatch(BATCHSIZE, 1000)

# loads pretrained model
#model = get_new_model("convnext_tiny", not_original=True)
#ckpt = torch.load('convnext_tiny_cvst_clean.pt', map_location='cpu') #['model']
                # print(ckpt.keys())
#ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
#ckpt = {k.replace('base_model.', ''): v for k, v in ckpt.items()}
#ckpt = {k.replace('se_', 'se_module.'): v for k, v in ckpt.items()}
#ckpt = {"".join(("model.", k)): v for k, v in ckpt.items()}

#model.load_state_dict(ckpt)
#print(model)

'''
function feeds the loaded model with data
Arguments: list[dict[image:tensor,label:str]], model
Return: list[dict[image:tensor,label:str, prediction:tensor]]
'''
def feedModel(samples, model):
    assert(0 < len(samples) < len(imageDataset))
    samplesWithPrediction = []
    for sample in samples:
        image, label = sample['image'], sample['label']
        prediction = model(image)
        #print(prediction.max(1)[1])
        sample['prediction'] = prediction
        samplesWithPrediction.append(sample)
    return samplesWithPrediction
        
#samplesWithPrediction = feedModel(samples)

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


#plt.figure()
#visualize(samples3dim)
#extractValuesFromDict(samples3dim, 'label')
#plt.axis('off')
#plt.ioff()
#plt.show()

# function that finds the top k predictions
def findMaxPredictions(samples, k:int):
    
    predictionsMax = []
    predictionsIndices = []
    
    for dictionary in samples:
        predictions = dictionary['prediction']
        tempPredictionsMax = []
        tempPredictionsIndices = []
        for i in range (0, k):
            maximums = []
            indices = []
            maximums = predictions.max().item()
            indices = predictions.argmax().item()
            tempPredictionsMax.append(maximums)
            tempPredictionsIndices.append(indices)
            predictions[0][indices] = - float('inf') # set probability of maximum to -inf to search for the next maximum
        predictionsMax.append(tempPredictionsMax)
        predictionsIndices.append(tempPredictionsIndices)
        
    return (predictionsMax, predictionsIndices)

# function that finds the labels to the top k predictions
def findLabels(samples, k:int):
    
    (predictionsMax, predictionsIndices) = findMaxPredictions(samples, k)
    allTopKLabels = []
    
    for i in range (0, len(samples)):
        topKLabels = []
        for j in range(0, k):
            topILabel = class_katalog.NAMES[predictionsIndices[i][j]]
            topKLabels.append(topILabel)
        allTopKLabels.append(topKLabels)
        
    return allTopKLabels

#print(findLabels(samplesWithPrediction, 10))

def errorFkt(text):

    with gr.Blocks() as demo:
        gr.Markdown(f'''{text}''')
        gr.Markdown('''Please restart the Program''')
    demo.launch()
