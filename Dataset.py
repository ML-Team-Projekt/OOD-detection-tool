import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from utilities import createAnnotation
from model_loader import get_new_model
import pandas as pd
from IPython.display import display
from PIL import Image 
import random
import numpy as np

IMAGESROOTDIR = 'NINCO_OOD_classes'

class ImageDataset(Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir
        createAnnotation(self.rootDir)
        self.annotation =  pd.read_csv('output.csv')


    def __getitem__(self, index):
        data_path = self.annotation.iloc[index,0]
        image = Image.open(data_path)
        label = self.annotation.iloc[index,1]
        return image, label

    def __len__(self):
        return len(self.annotation)

# instance of class ImageDataset
# contains all 765 images with their respective labels
imageDataset = ImageDataset(rootDir=IMAGESROOTDIR)

# Class which is used to rescale the image to a given size
# input: outputSize: int
# return: tuple(PIL Image, label)
class Rescale:
    def __init__(self, outputSize):
        self.outputSize = outputSize   
        
        
    def __calculateNewSize(self, size):
        initialWidth, initialHeight = size
        
        
        RATIO = initialWidth/self.outputSize
        newWidth = self.outputSize
        newHeight = initialHeight/RATIO
        
        return (round(newWidth), round(newHeight))  
        
    #sample data is a tuple(image, label)
    def __call__(self, sampleData):
        image, label = sampleData
        
        size = image.size
        
        newWidth, newHeight = self.__calculateNewSize(size)
        
        transformedImage = fn.resize(image, [newHeight, newWidth])
        
        return transformedImage,label
    
# Class which is used to center crop non quadratic images
# input: outputSize: int
# return: tuple(PIL Image, label)
class CenterCrop:
    def __init__(self, outputSize):
        self.outputSize = outputSize
        
    # creates a quadratic image
    def __call__(self, sampleData):
        image, label = sampleData
        
        width, height = image.size
        
        if (width != height or width != self.outputSize):
            centerCrop = torchvision.transforms.CenterCrop(self.outputSize)

            return centerCrop(image), label
        return image,label
    
# Constants for the size of the images
RESCALE = 240
CROP = 240
# objects for resizing
rescale = Rescale(RESCALE)
crop = CenterCrop(CROP)
composed = T.Compose([rescale, crop])

# given an Index returns the transformed Image
# input: Index: int
# return: tuple(PIL Image, label)
def transform(index):
    assert index <= len(imageDataset)
    tmp = composed(imageDataset[index])
    return tmp

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
        assert (0 < index <= self.datasetLength)
        self.index = index
        (picture, label) = transform(index)
        image = pilToTensor(picture)
        sample3dim = {'image' : image, 'label' : label}
        image = image.unsqueeze(0)
        sample = {'image': image, 'label': label}
        return sample, sample3dim
    
# Amount of random samples 
BATCHSIZE = 4

dataloader = DataLoader(len(imageDataset))

'''
function creates a random batch of data with a given size
Arguments: batchsize:int
Return: an array with a dict[image:label] 
'''
def createRandomBatch(batchsize):
    assert (0<batchsize <= len(imageDataset))
    batch = []
    batch3dim = []
    indexList = []
    for i in range(batchsize):
        index = random.randint(0,len(imageDataset))
        indexList.append(index)
        sample, sample3dim = dataloader[index]
        batch.append(sample)
        batch3dim.append(sample3dim)
    return batch, batch3dim, indexList

samples, samples3dim, indexList = createRandomBatch(BATCHSIZE)

# loads pretrained model
model = get_new_model("convnext_tiny", not_original=True)


'''
function feeds the loaded model with data
Arguments: list[dict[image:tensor,label:str]]
Return: None
'''
def feedModel(samples):
    assert(0<len(samples)<len(imageDataset))
    for sample in samples:
        image, label = sample['image'], sample['label']
        prediction = model(image)
        sample["prediction"]=prediction
    return samples
        
        
        
newSamples = feedModel(samples)

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


plt.figure()
visualize(samples3dim)
extractValuesFromDict(samples3dim, 'label')
plt.axis('off')
plt.ioff()
plt.show()








