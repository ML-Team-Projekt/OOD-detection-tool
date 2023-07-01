
import requests
import os



def createFolder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        


def downloadImages(urls, path):
    createFolder(path)  # create the folder

    for url in urls:
        imageName = url.split("/")[-1]
        imagePath = os.path.join(path, imageName)

        if not os.path.isfile(imagePath):  # ignore if already downloaded
            response=requests.get(url,stream=True)

            with open(imagePath,'wb') as outfile:
                outfile.write(response.content)

