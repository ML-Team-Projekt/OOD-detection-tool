#!/usr/bin/env python3
# coding: utf-8
# paths
import os
import random

import nltk
import numpy as np

nltk.download('wordnet')

from nltk.corpus import wordnet as wn
from myImageNetDict import getDictRep

import sys

sys.path.insert(0, '/home/lilly/miniconda3/lib/python3.10/site-packages')
import torch
from model_loader import get_new_model
import gradio as gr
from Dataset import imageDataset
import json
from threading import Thread
import time
from utilities import feedModel, findLabels, createRandomBatch
from jsonHelper import JsonHelper

random.seed(0)
np.random.seed(0)


class SPA_Interface():

    def __init__(self) -> None:
        # variables
        self.loggedIn = False
        self.askedAmount = False
        self.defaultBatchSize = 10
        self.index = 0
        self.uID = None
        self.modelName = "convnext_tiny"
        self.batchSize = 1
        self.topTenList = []
        self.finished = False
        self.batch = []
        self.labelList = []
        self.sourceList = []
        self.startPredownload = 0
        self.models = ["convnext_tiny", "convnext_small"]
        # function calls
        self.indexList, self.model = self.__loadDataFromModel(self.batchSize)
        self.jsonHandler = JsonHelper(self.batchSize)

    '''
    Takes an amount of images after an start image and downloads these
    '''

    def __predownloadImages(self, start, amount):
        for index in self.indexList[start:start + amount]:
            sample, sample3dim, source = imageDataset[index]
            location = "".join(source.split("/")[-1:])
            location = "imgBatch/"+location
            self.__createPredictions(location)
            print(location)
            self.sourceList.append(location)
            self.batch.append(location)
            label = sample['label']
            self.labelList.append(label)
            print("end predownload \n")

    '''
    Creates prediction for downloaded images'''

    def __createPredictions(self, sample):
        prediction = feedModel(sample, self.model)
        topTen = findLabels(prediction, 10)
        self.topTenList.append(topTen)

    def __loadDataFromModel(self, batchSize):
        indexList = createRandomBatch(batchSize, self.uID, imageDataset.datasetLength)

        if self.modelName == "convnext_tiny":
            model = get_new_model(self.modelName, not_original=True)
            ckpt = torch.load('convnext_tiny_cvst_clean.pt', map_location='cpu')
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
        if self.modelName == "convnext_small":
            model = get_new_model(self.modelName, pretrained=False, not_original=True)
            ckpt = torch.load('convnext_s_cvst_clean.pt', map_location='cpu')
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)

        if not os.path.exists('predictions'):
            os.makedirs('predictions')

        return indexList, model

    '''
    Find maximum predictions of Tensors
    '''

    def findMaxPred(self, prediction, k=10):
        predictionsMax = []
        predictionsIndices = []

        for i in range(0, k):
            maximums = prediction.max().item()
            indices = prediction.argmax().item()
            predictionsMax.append(maximums)
            predictionsIndices.append(indices)
            prediction[0][indices] = - float(
                'inf')  # set probability of maximum to -inf to search for the next maximum

        return (predictionsMax, predictionsIndices)

    '''
    gets the corresponding wordnet id from a given prediction
    '''

    def findWordNetID(self, prediction: str):
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

    '''
    fetches a description from the WordNet hierarchy for a given prediction
    '''

    def fetchSummaryFromNLTK(self, firstPrediction: str):
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
                print(hypernyms)
        else:
            summary = definition

        return summary

    '''
    Returns first label for new image
    '''

    def handleImageInput(self):
        print("handleImageInput")
        firstLabel = list(self.topTenList[self.index].keys())[0]
        summaryFirstLabel = self.fetchSummaryFromNLTK(firstLabel)
        return summaryFirstLabel

    '''
    Checks if user is already registered and assigns new Id
    '''

    def __authFunction(self, userInp):
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
                    self.jsonHandler.addUserId(self.uID)
                    return True
            if (len(emails_ids) == 0):
                newId = 1000
            else:
                newId = self.createNewUserID(emails_ids)
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
            self.jsonHandler.addUserId(self.uID)
            return True  # user has to get to know new Id
        else:
            for obj in emails_ids:
                if obj['userId'] == int(userInp):
                    self.uID = int(userInp)
                    self.jsonHandler.addUserId(self.uID)
                    return True
            return False  # user inputted a not existing Id

    '''
    Creates new UserID
    '''

    def createNewUserID(self, emails_ids):
        maxId = 1000
        for obj in emails_ids:
            if obj['userId'] > maxId:
                maxId = obj['userId']
        newId = maxId + 1
        return newId

    '''
    Increases index by 1
    '''

    def __incrementIndex(self):
        self.index += 1

    '''
    user selected decision (OOD/ IID/ abstain)
    '''

    def __selectDecision(self, decision: str):
        CONTINOUSDOWNLOAD = 10
        self.jsonHandler.addDecision(decision)
        if self.index >= self.batchSize - 1:
            return *[gr.update(visible=False) for _ in range(5)], *[gr.update(visible=True) for _ in
                                                                    range(3)], gr.update(
                value=f"Evaluation is over. Thank you for your help. Please remember your User ID: {self.uID}")

        else:
            print("select decision else \n")
            if (self.index % 10) == 3:
                self.__predownloadImages(self.startPredownload, CONTINOUSDOWNLOAD)
                self.startPredownload += CONTINOUSDOWNLOAD
            self.__incrementIndex()
            print(self.sourceList)
            return *[gr.update(visible=True) for _ in range(3)], gr.update(
                value=self.sourceList[self.index]), gr.update(value=self.topTenList[self.index]), *[
                gr.update(visible=False) for _ in range(3)], gr.update(value="")

    '''
    Handels login and prepares data for user
    '''

    def submitHandler(self, batchSize, userInput, model):

        self.loggedIn = self.__authFunction(userInput)
        if self.loggedIn:
            try:
                self.batchSize = int(batchSize)

            except:
                self.batchSize = self.defaultBatchSize

            finally:
                predownloadAmount = 4
                self.__predownloadImages(self.startPredownload, predownloadAmount)
                self.startPredownload += predownloadAmount
                self.jsonHandler.initData()
                print("start jsonHelper \n")
                self.modelName = model
                self.jsonHandler.addImgs(self.indexList, self.topTenList, self.modelName, self.sourceList, self.labelList)
                print("end jsonHelper \n")
                # self.addModel(model)
                return *[gr.update(visible=False) for _ in range(9)], *[gr.update(visible=True) for _ in
                                                                        range(3)], gr.update(visible=True,
                                                                                             value=f"Your model: {model}"), gr.update(
                    visible=True, value=f"Your User ID: {self.uID}"), gr.update(visible=True, value=self.sourceList[
                    self.index]), gr.update(visible=True, value=self.topTenList[self.index])

        else:
            try:
                int(userInput)
            except:  # user has to get to know new Id
                return *[gr.update(visible=True) for _ in range(7)], gr.update(visible=True,
                                                                               value=f"Your new Id: {self.uID}. Please insert it in the first input field and submit again."), *[
                    gr.update(visible=False) for _ in range(4)], gr.update(visible=True,
                                                                           value=f"Your new Id: {self.uID}. Please insert it in the first input field and submit again."), *[
                    gr.update(visible=False) for _ in range(2)], gr.update(
                    value=self.sourceList[self.index]), gr.update(value=self.topTenList[self.index])
            else:  # user inputted a not existing Id
                return *[gr.update(visible=True) for _ in range(6)], *[gr.update(visible=False) for _ in
                                                                       range(2)], gr.update(visible=True), *[
                    gr.update(visible=False) for _ in range(7)]

    '''
    Saves the decision data to our dataset
    '''

    def saveData(self):
        # add the new genarated data into database(here as dict)
        self.jsonHandler.updateData()

        databasePath = 'data.json'
        # write data into json file
        with open(databasePath, 'w') as database:
            json.dump(self.jsonHandler.getData(), database, indent=4)
        self.finished = True
        self.jsonHandler.setDataCollector(dict())
        self.jsonHandler.setDecisions([])

        return *[gr.update(visible=False) for _ in range(2)], *[gr.update(visible=False) for _ in range(3)]

    '''
    Saves user decisions and returns to first page
    '''

    def saveAndBack(self):
        self.jsonHandler.updateData()

        databasePath = 'data.json'
        with open(databasePath, 'w') as database:
            json.dump(self.jsonHandler.getData(), database, indent=4)
        self.jsonHandler.setDataCollector(dict())
        self.jsonHandler.setDecisions([])
        random.seed(0)
        np.random.seed(0)
        return *[gr.update(visible=True) for _ in range(6)], gr.update(value=self.uID), gr.update(
            value=self.batchSize), gr.update(value=self.modelName), *[gr.update(visible=False) for _ in range(3)]

    '''
    Ending page
    '''

    def lastPage(self):
        self.finished = True
        # empty the container
        self.jsonHandler.setDataCollector(dict())
        self.jsonHandler.setDecisions([])

        return *[gr.update(visible=False) for _ in range(2)], *[gr.update(visible=False) for _ in range(3)]

    '''
    Stops running threads
    '''

    def sysExit(self):
        while self.finished == False:
            time.sleep(1)
        thread_1._stop()

    def interface(self):
        with gr.Blocks() as demo:
            # login
            with gr.Row() as login1:
                text = gr.Markdown(value="Please insert your username or your User ID and your desired batchsize.")
            with gr.Row() as login2:
                username = gr.Textbox(label="Username or User ID", placeholder="Insert your username")
            with gr.Row() as login3:
                defaultBatchText = gr.Markdown(
                    "If you don't enter a batchsize, a batch of 10 samples will automatically be generated.")
            with gr.Row() as login4:
                batchSize = gr.Textbox(label="Batchsize",
                                       placeholder="Insert the amount of images you want to classify")

            with gr.Row() as login5:
                dropdown = gr.Dropdown(choices=self.models, value=self.modelName, interactive=True,
                                       label="Choose your model")

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
                    print(self.index)
                    image = gr.Image().style(height=400)
                    description = gr.Markdown(
                        value="")
                with gr.Column():
                    labels = gr.Label(label="Predictions", value="")

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

            submitButton.click(self.submitHandler, inputs=[batchSize, username, dropdown],
                               outputs=[login1, login2, login3, login4, login5, login6, newAcc, auth, IDinv,
                                        classifier1, classifier2, classifier3, choosenModel, userID, image, labels])
            buttonOOD.click(self.__selectDecision, inputs=decisionOOD,
                            outputs=[classifier1, classifier2, classifier3, image, labels, end1, end2, end3, text1])
            buttonIID.click(self.__selectDecision, inputs=decisionIID,
                            outputs=[classifier1, classifier2, classifier3, image, labels, end1, end2, end3, text1])
            buttonABS.click(self.__selectDecision, inputs=decisionAbstain,
                            outputs=[classifier1, classifier2, classifier3, image, labels, end1, end2, end3, text1])
            buttonBack.click(self.saveAndBack, inputs=None,
                             outputs=[login1, login2, login3, login4, login5, login6, username, batchSize, dropdown,
                                      end1, end2, end3])
            buttonConfirm.click(self.saveData, inputs=None, outputs=[classifier2, classifier3, end1, end2, end3])
            buttonClose.click(self.lastPage, inputs=None, outputs=[classifier2, classifier3, end1, end2, end3])

        demo.launch(inbrowser=True)


SPA = SPA_Interface()

thread_1 = Thread(target=SPA.interface)
thread_2 = Thread(target=SPA.sysExit)

thread_1.start()
thread_2.start()


