import requests
import os
from tqdm import tqdm


def createFolder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def downloadImages(urls, path):
    createFolder(path)  # create the folder

    for url in urls:
        imageName = url.split("/")[-1]
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
