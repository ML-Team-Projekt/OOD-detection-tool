#!/usr/bin/env python3
# coding: utf-8

import csv
import os
import random
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torchvision.transforms as T
import torchvision.transforms.functional as fn
from PIL import Image
from torch.utils.data import Dataset

random.seed(0)
np.random.seed(0)


class ImageDataset(Dataset):
    def __init__(self, annotation):
        self.annotation = pd.read_csv(annotation)
        self.batchFolder = 'imgBatch'
        if not os.path.exists(self.batchFolder):
            os.makedirs(self.batchFolder)

    def __getitem__(self, index):
        data_path = self.annotation.iloc[index, 0]
        image = self.__fetchOneImg(index, self.batchFolder)
        image = BytesIO(image)
        image = Image.open(image)
        label = self.annotation.iloc[index, 1]
        source = 'imgBatch/' + data_path.split('/')[-1]
        return image, label, source

    def __len__(self):
        return len(self.annotation)

    def __getRow(self, row_number, file_path='output.csv'):
        with open(file_path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

            assert not (row_number < 0 or row_number >= len(rows))
            row = rows[row_number]
            img = row['Data'].split('/')[-1]
            return {'label': row['Label'], 'img': img}

    # creates url for fetching images from server
    def __createUrl(self, row_number):
        dict = self.__getRow(row_number)
        imgName = dict['img']
        url = f"https://nc.mlcloud.uni-tuebingen.de/index.php/s/TgSK4n8ctPbWP4K/download?path=%2F{dict['label']}&files={dict['img']}"
        return url, imgName

    # fetches one image from the server in saves it to imgFolder
    def __fetchOneImg(self, imgIndex, imgFolder):
        url, img = self.__createUrl(imgIndex)
        response = requests.get(url)
        filename = imgFolder + '/' + img
        with open(filename, 'wb') as file:
            file.write(response.content)
        return response.content


# instance of class ImageDataset
# contains all 765 images with their respective labels
imageDataset = ImageDataset(annotation='output.csv')


# Class which is used to get the resized images with label
# input: datasetLength: int
# output:{'image': Tensor, 'label': String}
class DataLoader(Dataset):
    def __init__(self, datasetLength):
        self.datasetLength = datasetLength
        self.SIZE = round(224 / 0.875)

        # objects for tensor __transformation
        self.pilToTensor = T.ToTensor()
        self.tensorToPil = T.ToPILImage()

    def __getitem__(self, index):
        self.index = index
        (picture, label, source) = self.__transform(index)
        image = self.pilToTensor(picture)
        image = image.unsqueeze(0)
        sample = {'image': image, 'label': label}
        return sample, source

    # Constants for the size of the images

    # given an Index returns the __transformed Image
    # input: Index: int
    # return: tuple(PIL Image, label)
    def __transform(self, index):
        assert index <= len(imageDataset)
        FINAL_SIZE = 224
        image, label, source = imageDataset[index]
        rescaledImage = fn.resize(img=image, size=[self.SIZE, self.SIZE], interpolation=T.InterpolationMode.BICUBIC)
        transformedImage = fn.center_crop(img=rescaledImage, output_size=[FINAL_SIZE, FINAL_SIZE])
        return transformedImage, label, source


dataloader = DataLoader(len(imageDataset))
