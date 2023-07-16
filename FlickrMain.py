
import os
from FlickrAccess import getUrls
from Downloader import downloadImages
from FlickrClassifier import createFlickrData

allTags = ['baracuda', 'donut', 'shuttlecock']
imagesPerTag = 10


def download():
    for tag in allTags:

        print('Getting urls for', tag)
        urls = getUrls(tag, imagesPerTag)
        
        print('Downloading images for', tag)
        path = os.path.join('flickr-data', tag)

        downloadImages(urls, path)


download()
print('Done')

createFlickrData()



