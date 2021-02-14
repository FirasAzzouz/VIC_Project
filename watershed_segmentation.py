import sys, os
import numpy as np
import cv2 as cv
from skimage.segmentation import watershed as skimage_watershed
from sklearn.cluster import KMeans
from skimage.filters import sobel


class Watershed_Kmeans:
    def __init__(self, n_clusters=3, n_markers_foreground=100, n_markers_background=100, random_state=0):
        assert n_clusters>=2, "The number of clusters must be equal to or higher than 2"
        self.n_clusters = n_clusters
        self.n_markers_foreground = n_markers_foreground
        self.n_markers_background = n_markers_background
        self.random_state = random_state
        self.labels_kmeans =None
        self.markers = None

    def apply(self, img, markers_filter=None):
        # Apply kmeans to different regions of interest
        X = img.flatten().reshape((img.shape[0] * img.shape[1], 1))
        k = self.n_clusters
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        labels_kmeans = kmeans.labels_.reshape(img.shape)
        self.labels_kmeans = labels_kmeans

        # Compute the average lightness of each cluster
        img_avg = np.zeros(img.shape)
        avgs_unique = []
        for i in range(k):
            pos = np.where(labels_kmeans==i)
            avg = np.mean(img[pos])
            img_avg[pos] = avg
            avgs_unique.append(avg)

        # Get the cluster with the highest average lightness
        c = np.argmax(avgs_unique)

        # Consider this cluster as the foreground and the others as the background
        pos_foreground = np.where(labels_kmeans==c)
        pos_background = np.where(labels_kmeans!=c)

        # Sample points to indicate the foreground and the background
        markers = np.zeros_like(img)
        np.random.seed(self.random_state)
        markers_foreground = np.random.choice(range(len(pos_foreground[0])), self.n_markers_foreground, replace=False)
        markers_background = np.random.choice(range(len(pos_background[0])), self.n_markers_background, replace=False)

        # Assign the background markers (value 1) and the foreground markers (value 2)
        self.markers = np.zeros_like(img)
        for i in markers_background: self.markers[pos_background[0][i], pos_background[1][i]] = 1
        for i in markers_foreground: self.markers[pos_foreground[0][i], pos_foreground[1][i]] = 2

        # Apply a filter on the markers
        if markers_filter:
            self.markers = markers_filter(self.markers, img, n_markers=self.n_markers_foreground / 5)

        # Apply marker-controlled watershed on the elevation map
        elevation_map = sobel(img)
        labels = skimage_watershed(elevation_map, markers=self.markers)

        return labels
