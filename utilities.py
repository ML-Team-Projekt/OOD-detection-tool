import os
import csv
import requests

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
        img = row['Data'].split('\\')[-1]
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
