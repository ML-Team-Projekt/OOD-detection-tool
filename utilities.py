import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as fn
from PIL import Image

import class_katalog
from Dataset import imageDataset, dataloader
from model_loader import get_new_model

import requests
import json

import time

random.seed(0)
np.random.seed(0)

# from Dataset import imageDataset, dataloader


# Constants for the size of the images

SIZE = round(224 / 0.875)


def createAnnotation(folderPath):
    dataList = []
    labelList = []
    lengthOfFolderPath = len(folderPath)
    for root, dirs, files in os.walk(folderPath):
        for filename in files:
            dataList.append(os.path.join(root, filename))
            labelList.append(root[lengthOfFolderPath + 1:])
    totalNumberOfData = len(dataList)

    header = ['Data', 'Label']
    # open the file in the write mode
    with open("output.csv", mode="w", newline="", encoding="utf-8") as csvFile:
        # create the csv writer
        writer = csv.writer(csvFile)
        writer.writerow(header)
        for i in range(totalNumberOfData):
            writer.writerow([dataList[i], labelList[i]])


def createModel(modelName):
    model = None
    if modelName == "convnext_tiny":
        model = get_new_model(modelName, not_original=True)
        ckpt = torch.load('convnext_tiny_cvst_clean.pt', map_location='cpu')
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
    if modelName == "convnext_small":
        model = get_new_model(modelName, pretrained=False, not_original=True)
        ckpt = torch.load('convnext_s_cvst_clean.pt', map_location='cpu')
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
    return model


# creates list of the topK predictions
def createTopk(batch, model, amount, path='imgBatch/'):
    topTenList = []
    for img in batch:
        try:
            prediction = feedModel(img, model, path)
            topTen = findLabels(prediction[0], amount)
        except:
            print(f"        Couldn't create a prediction for {img}.")
            topTen = dict()
        finally:
            topTenList.append(topTen)
    return topTenList


# callable object to transform PIL to Tensor
pilToTensor = T.ToTensor()

'''
function feeds the loaded model with data
Arguments: list[dict[image:tensor,label:str]], model
Return: list[dict[image:tensor,label:str, prediction:tensor]]
'''
def feedModel(img, model, path):
    image = Image.open(path + img)
    image = fn.resize(img=image, size=[SIZE, SIZE], interpolation=T.InterpolationMode.BICUBIC)
    image = fn.center_crop(img=image, output_size=[SIZE, SIZE])
    image = pilToTensor(image)
    image = image.unsqueeze(0)

    prediction = model(image)
    prediction = F.softmax(prediction, dim=1)

    return prediction


# function that finds the labels to the top k predictions
def findLabels(prediction, k: int):
    (predictionMax, predictionIndices) = findMaxPredictions(prediction, k)

    topKLabels = {}
    for j in range(0, k):
        topILabel = class_katalog.NAMES[predictionIndices[j]]
        topKLabels[topILabel] = predictionMax[j]

    return topKLabels


# function that finds the top k predictions
def findMaxPredictions(prediction, k: int):
    tempPredictionsMax = []
    tempPredictionsIndices = []

    for _ in range(0, k):
        maximums = prediction.max().item()
        indices = prediction.argmax().item()
        tempPredictionsMax.append(maximums)
        tempPredictionsIndices.append(indices)
        prediction[indices] = - float('inf')  # set probability of maximum to -inf to search for the next maximum

    return (tempPredictionsMax, tempPredictionsIndices)


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
        RuntimeError(
            f"Your batch size {batchsize} is not in the range 0 < batch size < Länge von {IMAGESROOTDIR} = {len(imageDataset)}")
    batch = []
    indexList = []
    sourceList = []
    labelList = []
    attempts = 0

    iterator = 0
    with open('data.json', 'r') as file:
        saves = json.load(file)
        

    while iterator < batchsize:
        iterator += 1
        if attempts >= TRIALSTHRESHOLD:
            RuntimeError(
                f"The program tried more than {TRIALSTHRESHOLD} times to find an image which was not already shown to you. "
                f"Please try to enter a smaller amount of tries than {len(indexList)}.")

        flag = False
        index = random.randint(0, len(imageDataset))
        if index in indexList:
            iterator -= 1
            attempts += 0.5
            flag = True

        for img in saves:
            if uId != None and img['ImgID'] == index:
                user_calls = img['UserCall']
                for call in user_calls:
                    if call['userId'] == int(uId):
                        iterator -= 1
                        attempts += 1
                        flag = True

        if flag:
            continue

        indexList.append(index)
        sample, sample3dim, source = dataloader[index]

        sourceList.append(source)
        imgFile = source.split('/')[-1]
        batch.append(imgFile)
        label = sample['label']
        labelList.append(label)

    return batch, indexList, sourceList, labelList


'''
function extracts the values from the samples dict
Arguments: dict which contains random batch dict
Return: returns the values from samples list
'''
def extractValuesFromDict(samples, key: str):
    values = []
    for dictionary in samples:
        values.append(dictionary[key])
    return values


# function to visualize the batch
def visualize(samples):
    tensors = extractValuesFromDict(samples, 'image')
    grid_border_size = 2
    elementsPerRow = 4
    grid = torchvision.utils.make_grid(tensor=tensors, nrow=elementsPerRow, padding=grid_border_size)
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))



def fetchPredcitionForImage(image,model):
    PATH = f" https://nc.mlcloud.uni-tuebingen.de/index.php/s/J6TAAkfsdJzcGBR/download?path=&files=data.json"
    response = requests.get(PATH)
    
    if response.status_code == 200:
        jsonData = response.json()
    else:
        raise "ServerError, can't fetch file"
    
    for prediction in jsonData:
        if prediction['source'] == image:
            return prediction['topTen'][model]
        
    return None