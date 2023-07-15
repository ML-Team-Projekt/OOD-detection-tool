#!/usr/bin/env python3
# coding: utf-8


from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
import torchvision.transforms as T
from utilities import fetchOneImg
import pandas as pd
from PIL import Image
import random
import numpy as np
import os
from io import BytesIO

random.seed(0)
np.random.seed(0)


class ImageDataset(Dataset):
    def __init__(self, annotation):
        self.SIZE = round(224 / 0.875)
        self.annotation = pd.read_csv(annotation)
        self.batchFolder = 'imgBatch'
        if not os.path.exists('imgBatch'):
            os.makedirs('imgBatch')
        # objects for tensor transformation
        self.pilToTensor = T.ToTensor()
        self.tensorToPil = T.ToPILImage()
        self.datasetLength = len(self)

    def __getitem__(self, index):
        self.index = index
        (image, label, source) = self.transform(index)
        sample3dim = {'image': image, 'label': label}
        image = image.unsqueeze(0)
        sample = {'image': image, 'label': label}
        return sample, sample3dim, source

    def __fetchImageLabelAndSource(self, index):
        data_path = self.annotation.iloc[index, 0]
        image = fetchOneImg(index, self.batchFolder)
        image = BytesIO(image)
        image = Image.open(image)
        label = self.annotation.iloc[index, 1]
        source = data_path
        print(image)

        return image, label, source

    def __len__(self):
        return len(self.annotation)

    # given an Index returns the transformed Image
    # input: Index: int
    # return: tuple(PIL Image, label)
    def transform(self, index):
        # Constants for the size of the images

        assert index <= len(imageDataset)
        print("transform")
        image, label, source = self.__fetchImageLabelAndSource(index)
        print("t2 \n")
        rescaledImage = fn.resize(img=image, size=[self.SIZE, self.SIZE], interpolation=T.InterpolationMode.BICUBIC)
        transformedImage = fn.center_crop(img=rescaledImage, output_size=[self.SIZE, self.SIZE])
        transformedImage = self.pilToTensor(transformedImage)
        transformedImage = transformedImage.unsqueeze(0)
        return transformedImage, label, source


# instance of class ImageDataset
# contains all 765 images with their respective labels
imageDataset = ImageDataset(annotation='output.csv')
