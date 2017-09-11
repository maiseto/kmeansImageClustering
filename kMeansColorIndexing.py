import sys
from PIL import Image
import numpy as np
import math
import time
import random

MAX_ITERATIONS = 100
numFeatures = 3 # R, G, and B

# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(dataSet, k):
    # Initialize centroids randomly
    centroids = getRandomCentroids(k)

    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None

    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations):
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = centroids
        iterations += 1

        # Assign labels to each datapoint based on centroids
        # and get new centroids based on the average rgb value per centroid
        labels, centroids = getLabels(dataSet, centroids, k)
        print('labels: ', labels)
        print('centroids: ' ,centroids)
        # Assign centroids based on datapoint labels
        # centroids = getCentroids(dataSet, labels, k)

    # We can get the labels too by calling getLabels(dataSet, centroids)
    result = getPixelsFromLabel(labels, centroids)
    return result


# Function: Get Random Centroids
# -------------
# Return k array of randomized 3 values (R, G, B) between 0 to 255
# e.g. k=3: [[132, 142, 253], [171, 58, 150], [244, 45, 196]]
def getRandomCentroids(k):
    centroids = []
    for ID in range(0, k):
        try:
            pixel = random.sample(range(0, 255), numFeatures)
            centroids.append(pixel)
            # assign the random centroids to the centroidTable
        except ValueError:
            print('k exceeds number of pixels available')
    return centroids

# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def shouldStop(oldCentroids, centroids, iterations):
    if iterations > MAX_ITERATIONS: return True
    print('old: ', oldCentroids)
    print('new: ', centroids)
    return oldCentroids == centroids

# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset.
def getClosestCentroids(pixel, centroids):
    distToCentroid = []
    for cent in centroids:
        distToCentroid.append(int(calcDistance(cent, pixel)))

    closest = distToCentroid.index(min(distToCentroid))
    return closest

def calcDistance(centroid, pixel):
    r, g, b = pixel
    rc, gc, bc = centroid
    dist = math.sqrt((rc - r)**2 + (gc - g)**2 + (bc - b)**2)
    return dist

# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset.
# For each element in the dataset, chose the closest centroid.
# Make that centroid the element's label.
def getLabels(dataSet, centroids, k):
    arrLabels = []
    mean = [list([0,0,0]) for _ in range(k)]
    count = [0 for _ in range(k)]
    for i, row in enumerate(dataSet):
        rowLabels = []
        # centroidMean.append()
        for j, col in enumerate(row):
            closest = getClosestCentroids(col, centroids)
            rTotal, gTotal, bTotal = mean[closest]
            rTotal += col[0]
            gTotal += col[1]
            bTotal += col[2]
            mean[closest] = [rTotal, gTotal, bTotal]
            count[closest] += 1
            rowLabels.append(closest)
        arrLabels.append(rowLabels)
    # print('total: ',mean)
    # print('count: ',count)
    # print('---')
    for i, m in enumerate(mean):
        if count[i] == 0:
            centroids[i] = random.sample(range(0, 255), 3)
            getLabels(dataSet, centroids, k)
        else:
            mean[i] = [int(m[0]/count[i]), int(m[1]/count[i]), int(m[2]/count[i])]
    print(mean)
    return arrLabels, mean

# Function: Get Centroids
# -------------
# Returns k centroids, each of dimension n.
# Each centroid is the geometric mean of the points that
# have that centroid's label. Important: If a centroid is empty (no points have
# that centroid's label) you should randomly re-initialize it.
def getCentroids(dataSet, labels, k, centroidMean):
    arrCentroids = []
    return arrCentroids

def getPixelsFromLabel(labels, centroids):
    pixelsMatrix = list(labels)
    for i, row in enumerate(labels):
        for j, col in enumerate(row):
            pixelsMatrix[i][j] = centroids[col]
    return pixelsMatrix

# Function: Open Image
# -------------
# Converts PIL Image object from the file into a numpy array
def openImage(fileName):
    image = Image.open(fileName)
    arrPixels = np.array(image)
    return arrPixels

# Function: Get Centroids
# -------------
# Convert numpy array back as image file
def saveImage(fileName, npArray):
    image = Image.fromarray(npArray, 'RGB')
    image.save(fileName, quality=95)
    image.show()
    return

def main():
    start_time = time.time()
    npImage = openImage('cinque-terre-italy-cr-getty.jpg')
    result = np.asarray(kmeans(npImage, k=2))
    saveImage('cinque-terre-italy-cr-getty2.jpg', result.astype('uint8'))
    # for k in range(0, 8, 2):
    #     result = np.asarray(kmeans(npImage, k=2))
    #     fileName = 'Lennak' + str(k) + '.png'
    #     saveImage(fileName, result)
    print("--- %s seconds ---" % (time.time() - start_time))

    return

if  __name__ == '__main__':
    main()
