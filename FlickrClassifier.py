import json
import os
import time

import utilities

'''
Combines the other functions, controls the classification and data saving process of the flickr data
'''
def createFlickrData():
    startTime = time.time()
    print("\nStarting data creation process")
    imagePaths = getImagePaths('flickr-data')
    count = len(imagePaths)
    predictionsTiny, predictionsSmall = classifyFlickr(imagePaths)
    models = []
    labels = []
    print("     Sorting prediction data")
    for i in range(count):
        labels.append(imagePaths[i].split('/')[1])
        temData = {
            "convnext_tiny": predictionsTiny[i],
            "convnext_small": predictionsSmall[i]
        }
        models.append(temData)

    print("     Printing in File")
    for i in range(count):
        createJsonDatabase(labels[i], imagePaths[i], models[i])

    endTime = time.time()
    print(f"\nThe flickr data was classified and printed into 'flickrData.json'. The classification for {count} images"
          f" took {round(endTime - startTime)} seconds.")


'''
Collects alls the images with their individual paths of a root folder, and returns them as a list
'''
def getImagePaths(rootFolder):
    print("     Collecting image paths")
    imagePaths = []
    for root, dirs, files in os.walk(rootFolder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                imagePaths.append(file_path)

    print(f"        found {len(imagePaths)} images")
    return imagePaths


'''
Classifies our images with both models and returns the top 10 predictions
'''
def classifyFlickr(pics):
    print("     Starting classification")
    convnextTiny = utilities.createModel("convnext_tiny")
    convnextSmall = utilities.createModel("convnext_small")
    predictionsTiny = utilities.createTopk(pics, convnextTiny, amount=10, path="")
    predictionsSmall = utilities.createTopk(pics, convnextSmall, amount=10, path="")

    return predictionsTiny, predictionsSmall


'''
Saves out data in the database, ensures that there are no duplicates
'''
def createJsonDatabase(label, name, models):
    nameParts = name.split("-")
    data = {
        "name": name,
        "source": f"https://live.staticflickr.com/{nameParts[-2].split('/')[-1]}/{nameParts[-1]}",
        "label": label,
        "models": models
    }

    # Write JSON data to a file
    try:
        with open('flickrData.json', 'r') as file:
            existingData = json.load(file)
    except FileNotFoundError:
        existingData = []
    for entry in existingData:
        if entry['label'] == label and entry['name'] == name:
            # Duplicate entry found, do not append
            return

    existingData.append(data)

    with open('flickrData.json', 'w') as file:
        json.dump(existingData, file, indent=4)
