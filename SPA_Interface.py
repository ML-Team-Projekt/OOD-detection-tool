#!/usr/bin/env python3
# coding: utf-8
import random
#paths

import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn
from myImageNetDict import getDictRep

import sys 

sys.path.insert(0, '/home/lilly/miniconda3/lib/python3.10/site-packages')
import torch
import gradio as gr
#from Dataset import *
import json
import class_katalog
import os
from threading import Thread
import time
import shutil
from utilities import createModel, createTopk, createRandomBatch

import numpy as np
import random


random.seed(0)
np.random.seed(0)
class SPA_Interface():
    
    def __init__(self) -> None:
        self.loggedIn = False
        self.askedAmount = False
        self.defaultBatchSize = 10
        self.index = 0
        self.uID = None
        self.modelName = "convnext_tiny"
        self.batchSize = self.defaultBatchSize
        self.labels, self.indexList, self.sourceList, self.topTenList, self.labelList= self.__loadDataFromModel(self.batchSize)
        self.finished = False
        self.data = []
        self.decisions = []
        self.dataCollector = {}
        self.imgSet = set()
        
        self.models = ["convnext_tiny", "convnext_small"]
        
 
    # creates random batch and loads labels, indexList, sourceList, topTenList, labelList for the gui
    def __loadDataFromModel(self,batchSize):
        batch, indexList, sourceList, labelList = createRandomBatch(batchSize, self.uID)

        model = createModel(self.modelName)

        if not os.path.exists('predictions'):
            os.makedirs('predictions')
      
        topTenList = createTopk(batch, model, amount=10)
        labels = []
        
        for i in range(0, batchSize):
            labels.append(topTenList[i])

        return labels, indexList, sourceList, topTenList, labelList



    def initImgSet(self):
        if len(self.data) > 0:
            for obj in self.data:
                self.imgSet.add(obj['source'])

    def initData(self):
        # load existing data from database
        with open('data.json') as file:
            json_str = file.read()
        self.data = json.loads(json_str)
        self.initImgSet()

    def addImgs(self):
        folder_path = 'imgBatch/'
        shutil.rmtree(folder_path)
        os.makedirs('imgBatch')
        self.addBatchsize()
        self.labels, self.indexList, self.sourceList, self.topTenList, self.labelList= self.__loadDataFromModel(self.batchSize)
        self.addImg(self.indexList)
        self.addTopTen(self.topTenList)
        self.addModel(self.modelName)
        self.addSource(self.sourceList)
        self.addLabel(self.labelList)

    
    #Here we configure the Datacollector, which contains all informations of once Usercall
    def addUserId(self, UserId):
        self.dataCollector['UserId'] = UserId

    def addBatchsize(self):
        self.dataCollector['batchsize'] = int(self.batchSize)

    def addImg(self, imgIds):
        assert self.dataCollector['batchsize'] == len(imgIds)
        Samples = []
        for i in range(self.dataCollector['batchsize']):
            Samples.append({'ImgId': imgIds[i], 'topTen': []})
        self.dataCollector['Imgs'] = Samples

    def addTopTen(self, topTenList):
        assert len(topTenList) == len(self.dataCollector['Imgs'])
        for i in range(len(topTenList)):
            self.dataCollector['Imgs'][i]['topTen'] = topTenList[i]

    def addDecision(self, decision):
        self.decisions.append(decision)
        if len(self.decisions) == self.dataCollector['batchsize']:
            assert self.dataCollector['batchsize'] == len(self.dataCollector['Imgs'])
            for i in range(len(self.dataCollector['Imgs'])):
                self.dataCollector['Imgs'][i]['decision'] = self.decisions[i]
            self.decisions = []
        
    def addSource(self, sourceList):
        assert len(sourceList) == len(self.dataCollector['Imgs'])
        for i in range(len(self.dataCollector['Imgs'])):
            self.dataCollector['Imgs'][i]['source'] = sourceList[i]

    def addLabel(self, labelList):
        assert len(labelList) == len(self.dataCollector['Imgs'])
        for i in range(len(self.dataCollector['Imgs'])):
            self.dataCollector['Imgs'][i]['label'] = labelList[i]

    def addModel(self, model):
        self.dataCollector['model'] = model

    #create an object for a Usercall, in case that we want to insert a new image into our database
    def creatJsonObject(self, img):
        topTen = img['topTen']
        object = {
                    'ImgID' : img['ImgId'],
                    'source' : img['source'],
                    'label': img['label'],
                    'topTen': {self.dataCollector['model']: topTen},
                    'UserCall': [
                        {
                            'userId' : self.dataCollector['UserId'],
                            'model' : self.dataCollector['model'],
                            'decision': img['decision']
                        }
                    ]
                }

        return object

    #update the object of the image, which already exists in database.
    def updateObject(self, img):
        for obj in self.data:
            if obj['source'] == img['source']:
                call = {
                    'userId' : self.dataCollector['UserId'],
                    'model' : self.dataCollector['model'],
                    'decision': img['decision']
                }
                obj['UserCall'].append(call)
                #extend container of topTen if we call a new model for this image 
                key = self.dataCollector['model']
                if key not in obj['topTen']:
                    obj['topTen'][key] = img['topTen']

    #update our database everytime a Usercall happens
    def updateData(self):
        for img in self.dataCollector['Imgs']:
            if img['source'] not in self.imgSet:
                obj = self.creatJsonObject(img)
                self.data.append(obj)
                self.imgSet.add(img['source'])
            else:
                self.updateObject(img)

   
    
     # gets the corresponding wordnet id from a given prediction
    def findWordNetID(self,prediction:str):
        synsnetID = ""
        dictionaryMap = getDictRep()
        for _, value in dictionaryMap.items():
            foundID = value["id"]
            label = value["label"]
            
            if label.lower() == prediction.lower():
                
                trimmedID = foundID.split("-")[0]
                
                synsnetID = trimmedID
                break;

        return synsnetID
        
    
    # fetches a description from the WordNet hierarchy for a given prediction
    def fetchSummaryFromNLTK(self, firstPrediction:str):
        idForNLTK = self.findWordNetID(firstPrediction)
        summary = ""
        
        synset = wn.synset_from_pos_and_offset('n', offset=int(idForNLTK))

    
            
        definition = synset.definition()
        hypernyms = synset.hypernyms()
            
        if definition == "" or definition == None:
            while (hypernyms != None):
                synsnetHypernym = hypernyms[0]
                   
                hypernymDefinition = synsnetHypernym.definition()
                   
                if hypernymDefinition != "":
                    summary = hypernymDefinition
                    break
                hypernyms = synsnetHypernym.hypernyms()
        else:
            summary = definition

        return summary

    # extracts the firstLabel for the image and creates a nltk definition for the given image      
    def handleImageInput(self):
        firstLabel = list(self.topTenList[self.index].keys())[0]
        summaryFirstLabel = self.fetchSummaryFromNLTK(firstLabel)
        return summaryFirstLabel
        
     
    # authentication for the login process
    def __authFunction(self,userInp):
        # load existing emails and ID´s from database
        
        self.index = 0
        
        with open('emails_ids.json') as file:
            json_str = file.read()
        
        emails_ids = json.loads(json_str)
        
        try: 
            int(userInp)
        except:
            for obj in emails_ids:
                if obj['email'] == userInp:
                    self.uID = obj['userId']
                    self.addUserId(self.uID)
                    return True
            if (len(emails_ids) == 0):
                newId = 1000
            else:
                maxId = 1000
                for obj in emails_ids:
                    if obj['userId'] > maxId:
                        maxId = obj['userId']
                newId = maxId + 1
            dictIn = {
                'email': userInp,
                'userId': newId
            }
            emails_ids.append(dictIn)
            databasePath = 'emails_ids.json'
            # write email and new ID into json file
            with open(databasePath, 'w') as database:
                json.dump(emails_ids, database, indent=4)
            self.uID = newId
            self.addUserId(self.uID)
            return True # user has to get to know new Id
        else:
            for obj in emails_ids:
                if obj['userId'] == int(userInp):
                    self.uID = int(userInp)
                    self.addUserId(self.uID)
                    return True
            return False # user inputted a not existing Id
    
    # increments the index by 1
    def __incrementIndex(self):
        self.index +=1            
    
     # user selected decision (OOD/ IID/ abstain)
    def __selectDecision(self, decision:str):
        
        self.addDecision(decision)
        if self.index >= self.batchSize-1:
            return *[gr.update(visible=False) for _ in range(5)],*[gr.update(visible=True) for _ in range(3)], gr.update(value=f"Evaluation is over. Thank you for your help. Please remember your User ID: {self.uID}")
        else:
            self.__incrementIndex()
            return *[gr.update(visible=True) for _ in range(3)],gr.update(value=self.sourceList[self.index]),gr.update(value=self.topTenList[self.index]), *[gr.update(visible=False) for _ in range(3)], gr.update(value="")

    # event handler for the login submit process
    def submitHandler(self,batchSize, userInput, model):
        
        self.loggedIn = self.__authFunction(userInput)
        if self.loggedIn:
            try:            
                self.batchSize = int(batchSize)
                
            finally:
                self.initData()
                self.modelName = model
                self.addImgs()
        
                return *[gr.update(visible=False)  for _ in range(9)],*[gr.update(visible=True) for _ in range(3)], gr.update(visible=True, value=f"Your model: {model}"), gr.update(visible=True, value=f"Your User ID: {self.uID}"), gr.update(visible=True, value=self.sourceList[self.index]) ,gr.update(visible=True, value=self.topTenList[self.index])
    
        else:
            try:
                int(userInput)
            except: # user has to get to know new Id
                return *[gr.update(visible=True) for _ in range(7)], gr.update(visible=True, value=f"Your new Id: {self.uID}. Please insert it in the first input field and submit again."),*[gr.update(visible=False) for _ in range(4)], gr.update(visible=True, value=f"Your new Id: {self.uID}. Please insert it in the first input field and submit again."), *[gr.update(visible=False) for _ in range(2)], gr.update(value=self.sourceList[self.index]) ,gr.update(value=self.topTenList[self.index])
            else: # user inputted a not existing Id
                return *[gr.update(visible=True) for _ in range(6)],*[gr.update(visible=False) for _ in range(2)] ,gr.update(visible=True),*[gr.update(visible=False) for _ in range(7)]
    
    # saves the data to data.json after confirming
    def saveData(self):
        #add the new genarated data into database(here as dict)
        self.updateData()

        databasePath = 'data.json'
        # write data into json file
        with open(databasePath, 'w') as database:
            json.dump(self.data, database, indent=4)

        self.finished = True
        self.dataCollector = dict()
        self.decisions = []
        folder_path = 'imgBatch/'
        shutil.rmtree(folder_path)

        return *[gr.update(visible=False) for _ in range(2)], *[gr.update(visible=False) for _ in range(3)]
    
    # saves our data if user wants to classify more images
    def saveAndBack(self):
        self.updateData()

        databasePath = 'data.json'
        with open(databasePath, 'w') as database:
            json.dump(self.data, database, indent=4)
        self.dataCollector = dict()
        self.decisions = []
        random.seed(0)
        np.random.seed(0)
        return *[gr.update(visible=True) for _ in range(6)], gr.update(value=self.uID), gr.update(value=self.batchSize), gr.update(value=self.modelName), *[gr.update(visible=False) for _ in range(3)]


    # event handler for last page
    def lastPage(self):
        self.finished = True
        #empty the container
        self.dataCollector = dict()
        self.decisions = []

        return *[gr.update(visible=False) for _ in range(2)], *[gr.update(visible=False) for _ in range(3)]
    
    
    # kills our threads after the classification
    def sysExit(self):
        while self.finished == False:
            time.sleep(1)
        thread_1._stop()
        
    # gui for the classifier
    def interface(self):
        with gr.Blocks() as demo:
            
            # login
            with gr.Row() as login1:
                text= gr.Markdown(value="Please insert your username or your User ID and your desired batchsize.")
            with gr.Row() as login2:
                username = gr.Textbox(label="Username or User ID", placeholder="Insert your username")
            with gr.Row() as login3:
                defaultBatchText = gr.Markdown("If you don't enter a batchsize, a batch of 10 samples will automatically be generated.")
            with gr.Row()as login4:
                batchSize = gr.Textbox(label="Batchsize", placeholder="Insert the amount of images you want to classify")
                
            with gr.Row() as login5:
                dropdown = gr.Dropdown(choices=self.models, value=self.modelName, interactive=True, label="Choose your model")
                
            with gr.Row() as login6:
                submitButton = gr.Button("submit")
            with gr.Row(visible=True) as newAcc:
                auth = gr.Textbox(visible=False, value="")
            with gr.Row(visible=False) as IDinv:
                auth2 = gr.Textbox(visible=True, value="ID does not exist.")
        
            # image classifier
            with gr.Row(visible=False) as classifier1:
                choosenModel = gr.Markdown("")
                userID = gr.Markdown("Your User ID:")
            with gr.Row(visible=False) as classifier2:

                with gr.Column():
                    image = gr.Image(self.sourceList[self.index]).style(height=400)
                    description = gr.Markdown(value=f"{self.fetchSummaryFromNLTK(list(self.topTenList[self.index].keys())[0])}")
                with gr.Column():
                        
                    
                    labels = gr.Label(label="Top 10 Predictions",value=self.topTenList[self.index])
                   
                        
                       

            with gr.Row(visible=False) as classifier3:
                buttonOOD = gr.Button("OOD")
                buttonIID = gr.Button("IID")
                buttonABS = gr.Button("abstain")

            # for decision of user
            decisionOOD = gr.Textbox(visible=False, value="OOD")
            decisionIID = gr.Textbox(visible=False, value="IID")
            decisionAbstain = gr.Textbox(visible=False, value="Abstain")
                
            # end page
            with gr.Row(visible=False) as end1:
                text1 = gr.Markdown(value="")
            with gr.Row(visible=False) as end2:
                buttonBack = gr.Button("Label more pictures")
            with gr.Row(visible=False) as end3:
                buttonConfirm = gr.Button("Confirm and close")
                buttonClose = gr.Button("Close without confirming")
                
            image.change(self.handleImageInput, inputs=None, outputs=[description]) 
   
            submitButton.click(self.submitHandler, inputs=[batchSize, username, dropdown], outputs=[login1,login2,login3,login4,login5,login6,newAcc, auth,IDinv,classifier1,classifier2, classifier3, choosenModel, userID, image, labels])
            buttonOOD.click(self.__selectDecision, inputs=decisionOOD, outputs=[classifier1,classifier2, classifier3,image,labels,end1, end2, end3, text1])
            buttonIID.click(self.__selectDecision, inputs=decisionIID, outputs=[classifier1,classifier2, classifier3,image,labels,end1, end2, end3, text1])
            buttonABS.click(self.__selectDecision, inputs=decisionAbstain, outputs=[classifier1,classifier2, classifier3,image, labels,end1, end2, end3, text1])
            buttonBack.click(self.saveAndBack, inputs=None, outputs=[login1, login2, login3, login4, login5, login6, username, batchSize, dropdown, end1, end2, end3])
            buttonConfirm.click(self.saveData, inputs=None, outputs=[classifier2, classifier3, end1, end2, end3])
            buttonClose.click(self.lastPage, inputs=None, outputs=[classifier2,classifier3, end1, end2, end3])

        demo.launch(inbrowser=True)
        
SPA = SPA_Interface()

thread_1 = Thread(target=SPA.interface)
thread_2 = Thread(target=SPA.sysExit)

thread_1.start()
thread_2.start()
