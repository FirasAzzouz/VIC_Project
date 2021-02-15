import sys, os
import numpy as np
import matplotlib.pyplot as plt
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

    def apply(self, img, markers_filter=None, **kwargs):
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

        # Get the clusters with the highest average lightness
        foreground_clusters = np.argsort(-np.array(avgs_unique))[:2]
        c1, c2 = foreground_clusters
        # c = np.argmax(avgs_unique)

        # Consider these clusters as the foreground and the others as the background
        pos_foreground = np.where(np.logical_or(labels_kmeans==c1, labels_kmeans==c2))
        pos_background = np.where(np.logical_and(labels_kmeans!=c1, labels_kmeans!=c2))

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
            # self.markers = markers_filter(self.markers, img, n_markers=int(self.n_markers_foreground/10))
            self.markers = markers_filter(self.markers, img, **kwargs)


        # Apply marker-controlled watershed on the elevation map
        elevation_map = sobel(img)
        labels = skimage_watershed(elevation_map, markers=self.markers)

        return labels

class MRI_Segmentation:
    def __init__(self, dataset_path, filter_tumor=True):
        self.img_paths = []
        self.img_mask_paths = []
        self.dataset_path = dataset_path
        self.filter_tumor = filter_tumor
        self.scores = []

    def run(self, segmentation_method, nmax=-1, plot=False):
        i = 0
        for dirpath,_,filenames in os.walk(self.dataset_path):
            for f in filenames:
                if f[-3:] not in ["tif",]: continue
                if f[-8:-4]=="mask": continue
                f_mask = f[:-4] + "_mask" + ".tif"
                img_path = os.path.abspath(os.path.join(dirpath, f))
                img_mask_path = os.path.abspath(os.path.join(dirpath, f_mask))
                if self.filter_tumor:
                    if len(np.unique(cv.imread(img_mask_path))) < 2:continue

                self.img_paths.append(img_path)
                self.img_mask_paths.append(img_mask_path)

                img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
                img_mask = cv.imread(img_mask_path)[:, :, 0] / 255

                kwargs = {"n_markers" : 100}
                labels = segmentation_method.apply(img, MRI_Segmentation.markers_filter, **kwargs)

                score = MRI_Segmentation.evaluate(labels - 1, img_mask)
                self.scores.append(score)

                if plot:
                    print("Score = ", np.round(score, 2))
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    ax1.imshow(img, cmap="gray")
                    ax2.imshow(img_mask, cmap="gray")
                    ax3.imshow(labels, cmap="gray")
                    ax1.set_title("original image")
                    ax2.set_title("true mask")
                    ax3.set_title("watershed mask")
                    ax1.axis("off")
                    ax2.axis("off")
                    ax3.axis("off")
                    plt.show()

                i+=1
                if i>nmax: return

    @staticmethod
    def markers_filter(markers, img, n_markers=100):
        center = np.array([img.shape[0] / 2, img.shape[1] / 2])
        pos = np.where(markers==2)
        distance = np.array([np.linalg.norm(np.array([pos[0][i], pos[1][i]]) - center) for i in range(len(pos[0]))])
        index_sorted = np.argsort(distance)[n_markers:]
        background_x, background_y = pos[0][index_sorted], pos[1][index_sorted]
        markers_updated = np.copy(markers)
        markers_updated[background_x, background_y] = 0
        return markers_updated

    @staticmethod
    def evaluate(mask_1, mask_2):
        intersection = mask_1 * mask_2
        union = mask_1 + mask_2
        union[union > 1] = 1
        score = intersection.sum() / union.sum()
        return score
