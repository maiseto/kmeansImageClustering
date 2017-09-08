import numpy as np
import sys
import math
import random
from PIL import Image

MAX_ITERATIONS = 100

def openImage(filename):
    im = Image.open(filename)
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    #print(len(pixels[1]))
    #for p in pixels:
        #print(p)
    return pixels

# # Function: Should Stop
# # -------------
# # Returns True or False if k-means is done. K-means terminates either
# # because it has run a maximum number of iterations OR the centroids
# # stop changing.
# def shouldStop(oldCentroids, centroids, iterations):
#     if iterations > MAX_ITERATIONS:
#         return 1
#     # else, check for convergence and return
#     return oldCentroids == centroids
#
# # Function: Get Labels
# # -------------
# # Returns a label for each piece of data in the dataset.
# def getLabels(dataSet, centroids):
#     # For each element in the dataset, chose the closest centroid.
#     # Make that centroid the element's label.
#     return 0
#
# # Function: Get Centroids
# # -------------
# # Returns k random centroids, each of dimension n.
# def getCentroids(dataSet, labels, k):
#     # Each centroid is the geometric mean of the points that
#     # have that centroid's label. Important: If a centroid is empty (no points have
#     # that centroid's label) you should randomly re-initialize it.
#
#     return 0
#
# def getRandomCentroids(numFeatures, k):
#     try:
#         pixel = random.sample(range(0, 255), k)
#     except ValueError:
#         print('k exceeds number of pixels available')
#     return pixel
#
# def calcDistance(centroid, pixel):
#     r, g, b = pixel
#     rc, gc, bc = centroid
#     dist = math.sqrt((rc - r)**2 + (gc - g)**2 + (bc - b)**2)
#     return dist
#
# # Function: K Means
# # -------------
# # K-Means is an algorithm that takes in a dataset and a constant
# # k and returns k centroids (which define clusters of data in the
# # dataset which are similar to one another).
# def kmeans(dataSet, k):
#     # Initialize centroids randomly
#     numFeatures = 0 #dataSet.getNumFeatures()
#     centroids = getRandomCentroids(numFeatures, k)
#
#     # Initialize book keeping vars.
#     iterations = 0
#     oldCentroids = None
#
#     # Run the main k-means algorithm
#     while not shouldStop(oldCentroids, centroids, iterations):
#         # Save old centroids for convergence test. Book keeping.
#         oldCentroids = centroids
#         iterations += 1
#
#         # Assign labels to each datapoint based on centroids
#         labels = getLabels(dataSet, centroids)
#
#         # Assign centroids based on datapoint labels
#         centroids = getCentroids(dataSet, labels, k)
#
#     # We can get the labels too by calling getLabels(dataSet, centroids)
#     return centroids
#
#
# def main():
#     k = 2 # for now
#     pixels = openImage('Lenna.png')
#     kmeansRes = kmeans(pixels, k)
#     print(kmeansRes)
#
# if  __name__ == '__main__':
#     main()


