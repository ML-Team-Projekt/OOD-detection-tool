import os
import csv

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