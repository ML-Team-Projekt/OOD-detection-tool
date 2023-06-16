#!/usr/bin/env python3
# coding: utf-8

#paths 

import sys 

sys.path.insert(0, '/home/lilly/miniconda3/lib/python3.10/site-packages')
import torch
import wikipedia
from model_loader import get_new_model
import gradio as gr
from Dataset import *
import json
import class_katalog
from threading import Thread
import time

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
        self.decesions = []
        self.dataCollector = {}
        self.imgSet = set()

        
    def __loadDataFromModel(self,batchSize):
        batch, batch3dim, indexList, sourceList, labelList = createRandomBatch(batchSize, self.uID)
        model = get_new_model(self.modelName, not_original=True)
        if self.modelName == "convnext_tiny":
            ckpt = torch.load('convnext_tiny_cvst_clean.pt', map_location='cpu')
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
        samples = feedModel(batch, model) 
        topTenList = findLabels(samples, 10)
        labels = []
        # just for test purpose
        for i in range(0, batchSize):
            labels.append(topTenList[i])

        return labels, indexList, sourceList, topTenList, labelList
    
    def initImgSet(self):
        if len(self.data) > 0:
            for obj in self.data:
                self.imgSet.add(obj['source'])

    def initData(self):
        self.initImgSet()
        # load existing data from database
        with open('data.json') as file:
            json_str = file.read()

        self.data = json.loads(json_str)

    def addImgs(self):
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

    def addDecesion(self, decesion):
        self.decesions.append(decesion)
        if len(self.decesions) == self.dataCollector['batchsize']:
            assert self.dataCollector['batchsize'] == len(self.dataCollector['Imgs'])
            for i in range(len(self.dataCollector['Imgs'])):
                self.dataCollector['Imgs'][i]['decesion'] = self.decesions[i]
            self.decesions = []
        
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
                            'decesion': img['decesion']
                        }
                    ]
                }

        return object

    #update the Object of the image, which already exists in database.
    def updateObject(self, img):
        for obj in self.data:
            if obj['source'] == img['source']:
                call = {
                    'userId' : self.dataCollector['UserId'],
                    'model' : self.dataCollector['model'],
                    'decesion': img['decesion']
                }
                obj['UserCall'].append(call)
                #extend container of topTen if we call a new model for this image 
                key = self.dataCollector['model']
                if key not in obj['topTen']:
                    obj['topTen'][key] = img['topTen']

    #update our database ,everytime a Usercall happens
    def updateData(self):
        for img in self.dataCollector['Imgs']:
            if img['source'] not in self.imgSet:
                obj = self.creatJsonObject(img)
                self.data.append(obj)
                self.imgSet.add(img['source'])
            else:
                self.updateObject(img)

        
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
    
    def fetchSummaryFromWiki(self, pageTitle):
        try:
            pageTitle.capitalize()
            summary =  wikipedia.summary(pageTitle).split(".")[0]
        except wikipedia.exceptions.PageError:
            summary = ""
        return summary
      

    def handleImageInput(self):
        firstLabel = self.topTenList[self.index][0]
        restLabels = self.topTenList[self.index][1:-1]
        summaryFirstLabel = self.fetchSummaryFromWiki(firstLabel)
        firstLabel = firstLabel
        return firstLabel,summaryFirstLabel,restLabels
        
    def __authFunction(self,userInp):
        # load existing emails and Id´s from database
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
            # write email and new Id into json file
            with open(databasePath, 'w') as database:
                json.dump(emails_ids, database, indent=4)
            self.uID = newId
            return False # user has to get to know new Id
        else:
            for obj in emails_ids:
                if obj['userId'] == int(userInp):
                    self.uID = int(userInp)
                    self.addUserId(self.uID)
                    return True
            return False # user inputted a not existing Id
    
    def __incrementIndex(self):
        self.index +=1
        #if (self.index >= self.batchSize):
            #sys.exit(0)
            
    
    # user selected IID
    def __selectIID(self):#userId, iList, allL, bSize, loop):
        self.addDecesion("IID")
        if self.index >= self.batchSize-1:
            return gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True)
        else:
            self.__incrementIndex()
            return gr.update(value=self.sourceList[self.index]),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False)

    # user selected OOD    
    def __selectOOD(self):
        self.addDecesion("OOD")
        if self.index >= self.batchSize-1:
            return gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True)
        else:
            self.__incrementIndex()
            return gr.update(value=self.sourceList[self.index]),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False)

    # user selected abstinent
    def __selectAbstinent(self):
        self.addDecesion("Abstinent")
        if self.index >= self.batchSize-1:
            return gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True)
        else:
            self.__incrementIndex()
            return gr.update(value=self.sourceList[self.index]),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False)
    
    def submitHandler(self,batchSize, userInput):
        self.loggedIn = self.__authFunction(userInput)
        if self.loggedIn:
            try:            
                self.batchSize = int(batchSize)
            finally:
                self.initData()
                self.addImgs()
                return gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    
        else:
            try:
                int(userInput)
            except: # user has to get to know new Id
                return gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False), gr.update(visible=True, value=f"Your new Id: {self.uID}. Please insert it in the first input field and submit again."), gr.update(visible=False)
            else: # user inputted a not existing Id
                return gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
                
    
    def saveData(self):
        #add the new genarated data into database(here as dict)
        self.updateData()

        databasePath = 'data.json'
        # write data into json file
        with open(databasePath, 'w') as database:
            json.dump(self.data, database, indent=4)

        # update json
        return gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(value="Evaluation got saved. You can now exit the execution by clicking 'Close' once and then you can close the tab."),gr.update(visible=False),gr.update(visible=True)
    
    def lastPage(self):
        self.finished = True
        #empty the container
        self.dataCollector = dict()
        self.decesions = []

        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    def sysExit(self):
        while self.finished == False:
            time.sleep(1)
        thread_1._stop()
        
    def interface(self):
        with gr.Blocks() as demo:
            
            # login
            with gr.Row() as title:
                text= gr.Markdown(value="Please insert your username or your user ID and your desired batchsize. If you don´t enter a batchsize a batch of 10 samples will automatically be generated.")
            with gr.Row() as login1:
                username = gr.Textbox(label="Username or user ID")
            with gr.Row()as login2:
                batchSize = gr.Textbox(label="Batchsize")
            with gr.Row() as login3:
                submitButton = gr.Button("submit")
            with gr.Row(visible=True) as login4:
                auth = gr.Textbox(visible=False, value="")
            with gr.Row(visible=False) as login5:
                auth2 = gr.Textbox(value="ID does not exist.")
        
            # image classifier
            with gr.Row(visible=False) as classifier1:
                image = gr.Image(self.sourceList[self.index]).style(height=350)
            
            with gr.Row(visible=False) as classifier2: 
                labelOne = gr.Markdown(value=f"{self.topTenList[self.index][0]}")
                description = gr.Markdown(value=f"{self.fetchSummaryFromWiki(self.topTenList[self.index][0])}")
            with gr.Row(visible=False) as classifier3:
                labelRest = gr.Textbox(label="Other labels",value=", ".join(self.topTenList[self.index][1:-1]))

            with gr.Row(visible=False) as classifier4:
                buttonOOD = gr.Button("OOD")
                buttonIID = gr.Button("IID")
                buttonABS = gr.Button("abstinent")
                
            # end page
            with gr.Row(visible=False) as end1:
                text1 = gr.Markdown(value="Evaluation is over. Thanks for your help.")
            with gr.Row(visible=False) as end2:
                text2 = gr.Markdown(value="Please use the confirm-button to save your evaluation before you click 'Close'.")
                buttonConfirm = gr.Button("Confirm")
            with gr.Row(visible=False) as end3:
                buttonClose = gr.Button("Close")
                
            image.change(self.handleImageInput, inputs=None, outputs=[labelOne, description,labelRest]) 
            submitButton.click(self.submitHandler, inputs=[batchSize, username], outputs=[title,login1,login2,login3,classifier1,classifier2, classifier3, classifier4, auth, login5])
            buttonOOD.click(self.__selectOOD, inputs=None, outputs=[image,classifier2, classifier3, classifier4, end1, end2, end3])
            buttonIID.click(self.__selectIID, inputs=None, outputs=[image,classifier2, classifier3, classifier4, end1, end2, end3])
            buttonABS.click(self.__selectAbstinent, inputs=None, outputs=[image,classifier2, classifier3, classifier4, end1, end2, end3])
            buttonConfirm.click(self.saveData, inputs=None, outputs=[classifier1,classifier2, classifier3, classifier4, text1, end2, end3])
            buttonClose.click(self.lastPage, inputs=None, outputs=[classifier1,classifier2, classifier3, classifier4, end1, end2, end3])
        
        demo.launch()

SPA = SPA_Interface()

thread_1 = Thread(target=SPA.interface)
thread_2 = Thread(target=SPA.sysExit)

thread_1.start()
thread_2.start()

#SPA = SPA_Interface()
#SPA.interface()
