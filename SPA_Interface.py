#!/usr/bin/env python3
# coding: utf-8

#paths 

import sys 

sys.path.insert(0, '/home/lilly/miniconda3/lib/python3.10/site-packages')
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
import tqdm as notebook_tqdm
import gradio as gr
from Dataset import *
import json
import class_katalog

class SPA_Interface():
    
    def __init__(self) -> None:
        self.loggedIn = False
        self.askedAmount = False
        self.defaultBatchSize = 10
        self.index = 0
        self.uID = None
        self.batchSize = self.defaultBatchSize
        self.labels, self.indexList, self.sourceList, self.topTenList, self.labelList= self._loadDataFromModel(self.batchSize)
        
    def _loadDataFromModel(self,batchSize):
        batch, batch3dim, indexList, sourceList, labelList = createRandomBatch(batchSize)
        global modelName
        modelName = "convnext_tiny"
        model = get_new_model(modelName, not_original=True)
        if modelName == "convnext_tiny":
            ckpt = torch.load('convnext_tiny_cvst_clean.pt', map_location='cpu')
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
        samples = feedModel(batch) 
        topTenList = findLabels(samples, 10)
        labels = []
        # just for test purpose
        for i in range(0, batchSize):
            labels.append(topTenList[i])

        return labels, indexList, sourceList, topTenList, labelList
        
        
    def findMaxPred(self,prediction, k=10):
    
        predictionsMax = []
        predictionsIndices = []
        
        for i in range (0, k):
            maximums = []
            indices = []
            maximums = prediction.max().item()
            indices = prediction.argmax().item()
            predictionsMax.append(maximums)
            predictionsIndices.append(indices)
            prediction[0][indices] = - float('inf') # set probability of maximum to -inf to search for the next maximum                
        return (predictionsMax, predictionsIndices)

    def findLabels(self,sample, k=10):
        
        (predictionsMax, predictionsIndices) = self.findMaxPred(sample, k)
        allTopKLabels = []
    
        topKLabels = []
        for j in range (0, k):
            topILabel = []
            topILabel = class_katalog.NAMES[predictionsIndices[j]]
            topKLabels.append(topILabel)
        allTopKLabels.append(topKLabels)
            
        return allTopKLabels

    def handleImageInput(self):
        firstLabel = self.topTenList[self.index][0]
        restLabels = self.topTenList[self.index][1:-1]
        print(restLabels)
        return firstLabel,restLabels
        
    def __authFunction(self,userInp,passwordInp):
        users = {("testemail", "1001")}
        '''
            with open('emails_ids.json', 'r') as file:
                jsonData = json.load(file)
                for user in jsonData:
                    email, uId = user
                    print(type(email))
                    if user["email"] == userInp:
                        return True
        '''

        for user in users:
            email, userId = user
            if (email == userInp):
                self.uID = userId
                return True
            
        return False
    
    def __incrementIndex(self):
        self.index +=1
        if (self.index >= self.batchSize):
            sys.exit(0)
            
    
    # user selected IID
    def __selectIID(self):#userId, iList, allL, bSize, loop):
    # update json
        self.__incrementIndex()
        return gr.update(value=self.sourceList[self.index])

    # user selected OOD    
    def __selectOOD(self):
        self.__incrementIndex()
        return gr.update(value=self.sourceList[self.index])

    # user selected abstinent
    def __selectAbstinent(self):
        self.__incrementIndex()
        return gr.update(value=self.sourceList[self.index])
    
    def submitHandler(self,batchSize):
        if not self.loggedIn:
            self.loggedIn=True
            self.batchSize = int(batchSize)
            return gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True)
    
        
        return None
        
    def interface(self):
        with gr.Blocks() as demo:
            
            # login
            with gr.Row() as title:
                text= gr.Markdown(value="Please insert your username or your user ID and your desired batchsize")
            with gr.Row() as login1:
                username = gr.Textbox(label="Username")
            with gr.Row()as login2:
                batchSize = gr.Textbox(label="Batchsize")
            with gr.Row() as login3:
                submitButton = gr.Button("submit")
            
            
        
            # image classifier
            with gr.Row(visible=False) as classifier1:
                image = gr.Image(self.sourceList[self.index]).style(height=350)
            with gr.Row(visible=False) as classifier2:
                gr.Markdown("Test")
                labelOne = gr.Markdown(value=f"{self.topTenList[self.index][0]}")
                labelRest = gr.Textbox(value=self.topTenList[self.index][1:-1])

            with gr.Row(visible=False) as classifier3:
                buttonOOD = gr.Button("OOD")
                buttonIID = gr.Button("IID")
                buttonABS = gr.Button("abstinent")
                
                buttonOOD.click(self.__selectOOD, inputs=None, outputs=image)
                buttonIID.click(self.__selectIID, inputs=None, outputs=image)
                buttonABS.click(self.__selectAbstinent, inputs=None, outputs=image)
                
            image.change(self.handleImageInput, inputs=None, outputs=[labelOne, labelRest])
            submitButton.click(self.submitHandler, inputs=batchSize, outputs=[title,login1,login2,login3,classifier1,classifier2, classifier3])

        
        demo.launch()
    
    

SPA = SPA_Interface()
SPA.interface()