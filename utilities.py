import csv
import os

import requests
import torchvision.transforms as T
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import random
import class_katalog
import json
import gradio as gr
import numpy as np
# '''
# Creates the CSV
# '''
# def createAnnotation(folderPath):
#     dataList = []
#     labelList = []
#     lengthOfFolderPath = len(folderPath)
#     for root, dirs, files in os.walk(folderPath):
#         for filename in files:
#             dataList.append(os.path.join(root, filename))
#             labelList.append(root[lengthOfFolderPath+1:])
#     totalNumberOfData = len(dataList)
#
#     header = ['Data', 'Label']
#     # open the file in the write mode
#     with open("output.csv", mode="w", newline="", encoding="utf-8") as csvFile:
#         # create the csv writer
#         writer = csv.writer(csvFile)
#         writer.writerow(header)
#         for i in range(totalNumberOfData):
#             writer.writerow([dataList[i], labelList[i]])

'''
Reads our (CSV) data and returns it in a python dict
'''


def getRow(row_number, file_path='output.csv'):
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

        assert not (row_number < 0 or row_number >= len(rows))
        row = rows[row_number]
        img = row['Data'].split('/')[-1]
        return {'label': row['Label'], 'img': img}


'''
Creates Url for server connection
'''


def createUrl(row_number):
    dict = getRow(row_number)
    imgName = dict['img']
    url = f"https://nc.mlcloud.uni-tuebingen.de/index.php/s/TgSK4n8ctPbWP4K/download?path=%2F{dict['label']}&files={dict['img']}"
    return url, imgName


# downloads one image from server and saves to imgFolder
def fetchOneImg(imgIndex, imgFolder):
    print("fetchOneImg")
    url, img = createUrl(imgIndex)
    response = requests.get(url)
    filename = imgFolder + '/' + img
    with open(filename, 'wb') as file:
        file.write(response.content)

        print("Within write")

    print(os.listdir("./imgBatch"))
    return response.content


#
# def fetchBatch(indexList):
#     dirName = 'imgBatch'
#     print("fetchBatch")
#     if not os.path.exists(dirName):
#         os.makedirs(dirName)
#     for index in indexList:
#         fetchOneImg(index, dirName)
#
#
#
# def getFileNames(folder_path):
#     file_names = []
#     print("getFileName")
#     for file in os.listdir(folder_path):
#         if os.path.isfile(os.path.join(folder_path, file)):
#             file_names.append(file)
#     return file_names


'''
function creates a random batch of data with a given size
Arguments: batchsize:int
Return: an array with a dict[image:label] 
'''


def createRandomBatch(batchsize, uId, imageDatasetLength=0):
    # Number of tries to get another number
    random.seed(0)
    np.random.seed(0)
    TRIALSTHRESHOLD = 10000
    try:
        assert (0 < batchsize <= imageDatasetLength)
    except AssertionError:
        print("randombatcherror")#errorFkt(f"Your batch size {batchsize} is not in the range 0 < batch size < Länge von {imageDatasetLength}")
    indexList = []
    attempts = 0

    iteration = 0
    with open('data.json', 'r') as file:
        saves = json.load(file)
    while iteration < batchsize:
        iteration += 1
        if attempts >= TRIALSTHRESHOLD:
            print("randombatcherror")
            # errorFkt(
            #     f"The program tried more than {TRIALSTHRESHOLD} times to find an image which was not already shown to you. "
            #     f"Please try to enter a smaller amount of tries than {len(indexList)}.")
        alreadySeen = False
        index = random.randint(0, imageDatasetLength)
        if index in indexList:
            iteration -= 1
            attempts += 0.5
            alreadySeen = True

        for img in saves:
            if uId != None and img['ImgID'] == index:
                user_calls = img['UserCall']
                for call in user_calls:
                    if call['userId'] == int(uId):
                        iteration -= 1
                        attempts += 1
                        alreadySeen = True

        if alreadySeen:
            continue

        indexList.append(index)

    return indexList


'''
function extracts the values from the samples dict
Arguments: dict which contains random batch dict
Return: returns the values from samples list
'''


def extractValuesFromDict(samples, key: str):
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
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))


# function that finds the top k predictions
def findMaxPredictions(prediction, k: int):
    tempPredictionsMax = []
    tempPredictionsIndices = []

    for i in range(0, k):
        print(i)
        maximums = []
        indices = []
        maximums = prediction.max().item()
        indices = prediction.argmax().item()
        tempPredictionsMax.append(maximums)
        tempPredictionsIndices.append(indices)
        prediction[0][indices] = - float('inf')  # set probability of maximum to -inf to search for the next maximum

    print("\n")
    return (tempPredictionsMax, tempPredictionsIndices)


# function that finds the labels to the top k predictions
def findLabels(prediction, k: int):
    print(prediction)
    (predictionMax, predictionIndices) = findMaxPredictions(prediction, k)

    topKLabels = {}
    for j in range(0, k):
        topILabel = class_katalog.NAMES[predictionIndices[j]]
        topKLabels[topILabel] = predictionMax[j]

    return topKLabels


# def errorFkt(text):
#     with gr.Blocks() as demo:
#         gr.Markdown(f'''{text}''')
#         gr.Markdown('''Please restart the Program''')
#     demo.launch()


'''
    function feeds the loaded model with data
    Arguments: list[dict[image:tensor,label:str]], model
    Return: list[dict[image:tensor,label:str, prediction:tensor]]
    '''


def feedModel(img, model):
    image = Image.open(img)
    pilToTensor = T.ToTensor()
    image = pilToTensor(image)
    image = image.unsqueeze(0)
    prediction = model(image)
    prediction = F.softmax(prediction, dim=1)

    return prediction
