import os

from Downloader import downloadImages
from FlickrAccess import getUrls
from FlickrClassifier import createFlickrData

'''
Define the tags and the amount of pictures per tag
'''
ALLTAGS = ['baracuda', 'donut', 'shuttlecock']
IMAGESPERTAG = 10


'''
Downloads the images and amount defined above from the FlickrApi
'''
def download():
    for tag in ALLTAGS:
        print('Getting urls for', tag)
        urls = getUrls(tag, IMAGESPERTAG)

        print('Downloading images for', tag)
        path = os.path.join('flickr-data', tag)

        downloadImages(urls, path)

'''
Execute in order to start the download
'''
download()
print('Done')

'''
Execute in order to classify every image in flickr-data. This might take a while.
'''
createFlickrData()
