import os
import csv
import requests
import torch
from model_loader import get_new_model
import torch.nn.functional as F
import torchvision.transforms.functional as fn
import torchvision.transforms as T
from PIL import Image
import class_katalog


# Constants for the size of the images

SIZE = round(224/0.875)

def createAnnotation(folderPath):
    dataList = []
    labelList = []
    lengthOfFolderPath = len(folderPath)
    for root, dirs, files in os.walk(folderPath):
        for filename in files:
            dataList.append(os.path.join(root, filename))
            labelList.append(root[lengthOfFolderPath+1:])
    totalNumberOfData = len(dataList)

    header = ['Data', 'Label']
    # open the file in the write mode
    with open("output.csv", mode="w", newline="", encoding="utf-8") as csvFile:
        # create the csv writer
        writer = csv.writer(csvFile)
        writer.writerow(header)
        for i in range(totalNumberOfData):
            writer.writerow([dataList[i], labelList[i]])

def getRow(row_number, file_path = 'output.csv'):
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

        assert not (row_number < 0 or row_number >= len(rows))
        row = rows[row_number]
        img = row['Data'].split('/')[-1]
        return {'label':row['Label'], 'img':img}

def createUrl(row_number):
    dict = getRow(row_number)
    imgName = dict['img']
    url = f"https://nc.mlcloud.uni-tuebingen.de/index.php/s/TgSK4n8ctPbWP4K/download?path=%2F{dict['label']}&files={dict['img']}"
    return url, imgName

def fetchOneImg(imgIndex, imgFolder):
    url, img = createUrl(imgIndex)
    response = requests.get(url)
    filename = imgFolder + '/' + img
    with open(filename, 'wb') as file:
         file.write(response.content)
    return response.content

def fetchBatch(indexList):
    dirName = 'imgBatch'
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    for index in indexList:
        fetchOneImg(index, dirName)
    
def getFileNames(folder_path):
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_names.append(file)
    return file_names


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


pilToTensor = T.ToTensor()


'''
function feeds the loaded model with data
Arguments: list[dict[image:tensor,label:str]], model
Return: list[dict[image:tensor,label:str, prediction:tensor]]
'''
def feedModel(img, model, path):

    image = Image.open(path + img)
    image = fn.resize(img=image, size=[SIZE, SIZE], interpolation=T.InterpolationMode.BICUBIC)
    image = fn.center_crop(img=image, output_size=[SIZE,SIZE])
    image = pilToTensor(image)
    image = image.unsqueeze(0)

    prediction = model(image)
    prediction = F.softmax(prediction, dim=1)

    return prediction


# function that finds the labels to the top k predictions
def findLabels(prediction, k:int):
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

    for i in range(0, k):
        maximums = []
        indices = []
        maximums = prediction.max().item()
        indices = prediction.argmax().item()
        tempPredictionsMax.append(maximums)
        tempPredictionsIndices.append(indices)
        prediction[indices] = - float('inf')  # set probability of maximum to -inf to search for the next maximum

    return (tempPredictionsMax, tempPredictionsIndices)