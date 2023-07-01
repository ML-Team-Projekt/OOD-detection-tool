
from flickrapi import FlickrAPI

KEY = '50c7d487720201a7e74b98f517058eec'
SECRET = 'd45273e1b393ef1a'

# limiting the sizes we are interested in
flickr = FlickrAPI(KEY, SECRET, format='parsed-json')
SIZES = ["url_o", "url_k", "url_h", "url_l", "url_c"]

def getPhotos(imageTag):
    # All the extra data that we want to have
    extras = 'owner_name,description, url_o, url_k, url_h, url_l, url_c'

    photos = flickr.photos.search(
        tag=imageTag,  # Search term
        per_page=50,  # Number of results per page
        license='4,5,6,7,8,9,10',  # Attribution Licenses
        extras=extras,
        privacy_filter=1,  # public photos
        safe_search=1  # is safe
    )
    return photos


def getUrl(photo):
    for i in range(len(SIZES)):  # makes sure we search the sizes in the order above
        url = photo.get(SIZES[i])
        if url:  # if url is None try with the next size
            return url


# unites getPhotos and getUrls, returns list of URLs
def getUrls(imageTag, max):
    data = getPhotos(imageTag)
    pics = data['photos']['photo']
    counter=0
    urls=[]


    for pic in pics:
        if counter < max:
            url = getUrl(pic)  # get preffered size url
            if url:
                urls.append(url)
                counter += 1
        else:
            break

    return urls
