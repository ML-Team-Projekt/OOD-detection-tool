import os

import requests
from tqdm import tqdm

'''
Ensures that the given path exists
'''
def createFolder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

'''
Downloads every given urls given as a list, saves them at given path
'''
def downloadImages(urls, path):
    createFolder(path)  # create the folder

    for url in urls:
        urlParts = url.split("/")
        fileName = urlParts[-1]
        photoId = urlParts[-2]
        imageName = f"{photoId}-{fileName}"
        imagePath = os.path.join(path, imageName)

        if not os.path.isfile(imagePath):  # ignore if already downloaded
            response = requests.get(url, stream=True)
            totalSize = int(response.headers.get('content-length', 0))

            with open(imagePath, 'wb') as outfile, tqdm(
                    desc=imageName,
                    total=totalSize,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    outfile.write(data)
                    progress_bar.update(len(data))
