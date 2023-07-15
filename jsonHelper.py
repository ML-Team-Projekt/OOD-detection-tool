import json
import os
import shutil


class JsonHelper:
    def __init__(self, batchSize):
        self.__data = []
        self.__decisions = []
        self.__dataCollector = dict()
        self.__imgSet = set()
        self.__batchSize = batchSize

    '''
    Setter and Getter for our variables
    '''

    def setBatchSize(self, batchSize):
        self.__batchSize = batchSize

    def getBatchSize(self):
        return self.__batchSize

    def setData(self, data):
        self.__data = data

    def getData(self):
        return self.__data

    def setDecisions(self, decisions):
        self.__decisions = decisions

    def getDecisions(self):
        return self.__decisions

    def setImgSet(self, imgSet):
        self.__imgSet = imgSet

    def getImgSet(self):
        return self.__imgSet

    def setDataCollector(self, dataCollector):
        self.__dataCollector = dataCollector

    def getDataCollector(self):
        return self.__dataCollector

    def initImgSet(self):
        if len(self.__data) > 0:
            for obj in self.__data:
                self.__imgSet.add(obj['source'])

    def initData(self):
        # load existing data from database
        with open('data.json') as file:
            json_str = file.read()
        self.__data = json.loads(json_str)
        self.initImgSet()

    def addImgs(self, indexList, topTenList, modelName, sourceList, labelList):
        folder_path = 'imgBatch/'
        shutil.rmtree(folder_path)
        os.makedirs('imgBatch')
        self.addBatchsize()
        self.addImg(indexList)
        self.addTopTen(topTenList)
        self.addModel(modelName)
        self.addSource(sourceList)
        self.addLabel(labelList)

    # Here we configure the Datacollector, which contains all informations of once Usercall
    def addUserId(self, UserId):
        self.__dataCollector['UserId'] = UserId

    def addBatchsize(self):
        self.__dataCollector['batchsize'] = int(self.__batchSize)

    def addImg(self, imgIds):
        assert self.__dataCollector['batchsize'] == len(imgIds)
        Samples = []
        for i in range(self.__dataCollector['batchsize']):
            Samples.append({'ImgId': imgIds[i], 'topTen': []})
        self.__dataCollector['Imgs'] = Samples

    def addTopTen(self, topTenList):
        #assert len(topTenList) == len(self.__dataCollector['Imgs'])
        for i in range(len(topTenList)):
            self.__dataCollector['Imgs'][i]['topTen'] = topTenList[i]

    def addDecision(self, decision):
        self.__decisions.append(decision)
        if len(self.__decisions) == self.__dataCollector['batchsize']:
            assert self.__dataCollector['batchsize'] == len(self.__dataCollector['Imgs'])
            for i in range(len(self.__dataCollector['Imgs'])):
                self.__dataCollector['Imgs'][i]['decision'] = self.__decisions[i]
            self.__decisions = []

    def addSource(self, sourceList):
       # assert len(sourceList) == len(self.__dataCollector['Imgs'])
        for i in range(len(self.__dataCollector['Imgs'])):
            self.__dataCollector['Imgs'][i]['source'] = sourceList[i]

    def addLabel(self, labelList):
        #assert len(labelList) == len(self.__dataCollector['Imgs'])
        for i in range(len(self.__dataCollector['Imgs'])):
            self.__dataCollector['Imgs'][i]['label'] = labelList[i]

    def addModel(self, model):
        self.__dataCollector['model'] = model

    # create an object for a Usercall, in case that we want to insert a new image into our database
    def creatJsonObject(self, img):
        topTen = img['topTen']
        object = {
            'ImgID': img['ImgId'],
            'source': img['source'],
            'label': img['label'],
            'topTen': {self.__dataCollector['model']: topTen},
            'UserCall': [
                {
                    'userId': self.__dataCollector['UserId'],
                    'model': self.__dataCollector['model'],
                    'decision': img['decision']
                }
            ]
        }

        return object

    # update the object of the image, which already exists in database.
    def updateObject(self, img):
        for obj in self.__data:
            if obj['source'] == img['source']:
                call = {
                    'userId': self.__dataCollector['UserId'],
                    'model': self.__dataCollector['model'],
                    'decision': img['decision']
                }
                obj['UserCall'].append(call)
                # extend container of topTen if we call a new model for this image
                key = self.__dataCollector['model']
                if key not in obj['topTen']:
                    obj['topTen'][key] = img['topTen']

    # update our database everytime a Usercall happens
    def updateData(self):
        for img in self.__dataCollector['Imgs']:
            if img['source'] not in self.__imgSet:
                obj = self.creatJsonObject(img)
                self.__data.append(obj)
                self.__imgSet.add(img['source'])
            else:
                self.updateObject(img)
