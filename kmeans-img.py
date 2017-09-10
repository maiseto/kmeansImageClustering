import random
import math
import numpy as np
import sys
from PIL import Image
from collections import defaultdict
import time

MAX_ITERATIONS = 100
# this dict has the list of different centroid IDs and their centroid rgb values
centroidTable = {}
# this dict has the list of centroid IDs and their assigned pixel rgb values
centroidDB = defaultdict(list)

def openImage(filename):
    im = Image.open(filename)
    # print(np.array(im))
    oneLinerPixels = list(im.getdata())
    width, height = im.size
    pixels = [oneLinerPixels[i * width:(i + 1) * width] for i in range(height)]
    #print(len(pixels[1]))
    #for p in pixels:
        #print(p)
    return pixels, im

# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def shouldStop(converged, iterations):
    if iterations > MAX_ITERATIONS or converged == 1:
        return 1
    return 0

# Function: Get Centroid ID (key)
# -------------
# Returns the ID/key of a centroid
def getCentroidID(centroidValue):
    # print(centroidDB.items())
    for key, value in centroidDB.items():
        for v in value:
            # print('value        : ', v)
            # print('centroidValue: ', centroidValue)
            if v == centroidValue:
                return key
    print('cannot find')
    return -1

# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset.
def getLabels(dataSet):
    # For each element in the dataset, chose the closest centroid.
    # Make that centroid the element's label.
    centroidDB = defaultdict(list)
    for line in dataSet:
        for pixel in line:
            distToCentroid = []
            for cent in centroidTable.values():
                distToCentroid.append(calcDistance(cent, pixel))

            closest = distToCentroid.index(min(distToCentroid))
            # closestCentroid = centroidTable[closest]
            # print('closest cent: ', closestCentroid, ' pixel val: ', pixel)
            centroidDB[closest].append(pixel)
    #print(centroidDB)
    return

# Function: Get Centroids
# -------------
# Returns k random centroids, each of dimension n.
def getCentroids(dataSet, k):
    # print('before: ', centroidTable)
    oldCentroidTable = {key: value[:] for key, value in centroidTable.items()}

    # Each centroid is the geometric mean of the points that
    # have that centroid's label. Important: If a centroid is empty (no points have
    # that centroid's label) you should randomly re-initialize it.
    rTotal = 0
    gTotal = 0
    bTotal = 0
    newCentroids = []
    for key, value in centroidDB.items():
        for rgb in value:
            r, g, b = rgb
            rTotal += r
            gTotal += g
            bTotal += b
        clusterLength = len(centroidDB[key])
        # print('rtot: ', rTotal, ' gtot: ', gTotal, ' btot: ', bTotal)
        # print('cluster num: ', key, ' cluster length: ', len(centroidDB[key]))
        rgbMean = [int(rTotal / clusterLength), int(gTotal / clusterLength), int(bTotal/ clusterLength)]
        centroidTable[key] = rgbMean
        rTotal = 0
        gTotal = 0
        bTotal = 0
    # print('after: ', centroidTable)
    return oldCentroidTable == centroidTable


def getRandomCentroids(numFeatures, k):
    centroids = []
    for ID in range(0, k):
        try:
            pixel = random.sample(range(0, 255), numFeatures)
            centroids.append(pixel)
            # assign the random centroids to the centroidTable
            centroidTable[ID] = pixel
        except ValueError:
            print('k exceeds number of pixels available')
    return centroids

def calcDistance(centroid, pixel):
    r, g, b = pixel
    rc, gc, bc = centroid
    dist = math.sqrt((rc - r)**2 + (gc - g)**2 + (bc - b)**2)
    return dist

# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(dataSet, k):
    # Initialize centroids randomly
    numFeatures = 3 # RGB
    getRandomCentroids(numFeatures, k)

    # print('centroids: ', centroids)
    # Initialize book keeping vars.
    iterations = 0
    converged = 0
    # Run the main k-means algorithm
    while not shouldStop(converged, iterations):
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = centroidTable
        # print(oldCentroids)
        iterations += 1

        # Assign labels to each datapoint based on centroids
        getLabels(dataSet)
        #
        # print('len centdb ', len(centroidDB))
        # making sure there's no cluster with 0 elements inside
        count = 0
        # while iteration hasn't finished
        while count < len(centroidDB):
            print('len ', len(centroidDB[count]))
            if len(centroidDB[count]) == 0:
                getCentroids(dataSet, k)
                count = 0
                # start over
                break
            count += 1

        # Assign centroids based on datapoint labels
        converged = getCentroids(dataSet, k)
        # print('converged? ', converged)
    # We can get the labels too by calling getLabels(dataSet, centroids)
    return

def printImage(dataSet):
    newImage = []
    for line in dataSet:
        newImagePerLine = []
        for pixel in line:
            centroidID = getCentroidID(pixel)
            # print('centroidID: ', centroidID)
            newPixelValue = centroidTable[centroidID]
            # print('newPixelValue: ', newPixelValue)
            newImagePerLine.append(tuple(newPixelValue))

        newImage.append(newImagePerLine)
    return newImage

def main():
    start_time = time.time()
    k = 2 # for now
    pixels, pilImg = openImage('cubs.jpg')
    # print(pixels)
    kmeans(pixels, k)
    image = np.asarray(printImage(pixels))
    # print(len(image))
    # print(image)
    newPILImage = Image.fromarray(image.astype('uint8'))
    newPILImage.save('cubsk2v2.jpg')
    print("--- %s seconds ---" % (time.time() - start_time))

if  __name__ == '__main__':
    main()


