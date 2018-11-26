from __future__ import division
import cv2
import time
import logging

from utils import *
from image_manager import ImageManager
from match_geometric_filters import HistogramLogicFilter, RansacFilter
from data_plotter import DataPlotter


logging.basicConfig(level=logging.WARNING)


############################################
#              HYPERPARAMETERS             #
############################################


class AlignmentParameters:
    def __init__(self):
        self.threshold_angle = 0
        self.threshold_length = 0
        self.knn_neighbors = 0
        self.shrink_x_ratio = 0
        self.shrink_y_ratio = 0
        self.plot_images = False

        self.histogram_weight = None

        self.crop_iterations = 1


parameters = AlignmentParameters()

# Accepted std deviations from average
parameters.angle_th = 0.7     # Angular distribution
parameters.length_th = 0.7    # Length distribution

# Knn neighbors used. Cannot be changed from 2 right now
parameters.knn_neighbors = 2

# Image shrink factors in both dimensions
parameters.shrink_x_ratio = 1/4
parameters.shrink_y_ratio = 1/4

parameters.plot_images = True


# Knn weight ratio exploration. Relates how bigger must the first match be wrt the second to be considered a match
# parameters.histogram_weigh = np.arange(1.9, 1.3, -0.05)
parameters.histogram_weight = [1.4]


# Crop iterations counter. During each iteration the area of matching is reduced based on the most likely
# region of last iteration
parameters.crop_iterations = 1

# xIn = float(sys.argv[1])
# yIn = float(sys.argv[2])
# Img1path = sys.argv[3]
# Img2path = sys.argv[4]

train_image_path = 'Images/IMG_0568.JPG'
query_image_path = 'Images/IMG_0570.JPG'

# Img1path = 'Images/IMG_0596.JPG'
# Img2path = 'Images/IMG_0621.JPG'


############################################
#                MAIN BODY                 #
############################################

original_query_image = ImageManager()
original_train_image = ImageManager()

# Read images
start = time.time()
original_query_image.read_image(query_image_path)
mid = time.time()
original_train_image.read_image(train_image_path)
end = time.time()

print("TIME: Read query image. Elapsed time: ", mid - start)
print("TIME: Read train image. Elapsed time: ", end - mid)

original_data_plotter = DataPlotter(original_train_image, original_query_image)

# Generate image copies for the processing
query_image = ImageManager()
query_image.load_image(original_query_image.image)
start = time.time()
query_image.downsample(parameters.shrink_x_ratio, parameters.shrink_y_ratio)

train_image = ImageManager()
train_image.load_image(original_train_image.image)
mid = time.time()
train_image.downsample(parameters.shrink_x_ratio, parameters.shrink_y_ratio)
end = time.time()

print("TIME: Downsample query image. Elapsed time: ", mid - start)
print("TIME: Downsample train image. Elapsed time: ", end - mid)

processed_data_plotter = DataPlotter(train_image, query_image)

# Initialize shrink compensator
shrink_ratio = [1/parameters.shrink_x_ratio, 1/parameters.shrink_y_ratio]

# Initiate BoundingBox coordinates (top left corner, lower right corner). Only used for the snip matching algorithm
bb = np.array([[0, 0], [0, 0]])

# Initiate crop coordinate compensator. Only useful if crop_it > 1. It keeps track of sequential crops to transform
# crop coordinates into original image coordinates
cum_crop = [0, 0]

# Initiate the feature detector
cv2_detector = cv2.ORB_create()

# Find the key points and descriptors for train image
start = time.time()
train_image.find_keypoints(cv2_detector)
end = time.time()
print("TIME: Extract features of train image. Elapsed time: ", end - start)

# Instantiate histogram logic filter
histogram_filter = HistogramLogicFilter()


for it in range(0, parameters.crop_iterations):

    # Find the key points and descriptors from query image
    start = time.time()
    query_image.find_keypoints(cv2_detector)  # 2
    end = time.time()
    print("TIME: Extract features of query image. Elapsed time: ", end - start)

    # BFMatcher with default params, to find equivalent features
    # bf = cv2.BFMatcher()

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    start = time.time()
    matches = bf.match(train_image.descriptors, query_image.descriptors)
    end = time.time()
    print("TIME: Matching done. Elapsed time: ", end - start)

    # Initialize fitness trackers
    fitness = float('-inf')
    maxFit = fitness
    maxWeight = parameters.histogram_weight[0]

    # Explore all the weight values
    for weight in parameters.histogram_weight:

        # Filter knn matches by best to second best match ratio
        start = time.time()
        [good_matches, good_matches_pairs] = knn_match_filter(matches, weight)
        end = time.time()
        print("TIME: Distance filtering of matches done. Elapsed time: ", end - start)

        start = time.time()
        h_matrix = RansacFilter.ransac_homography(train_image.keypoints, query_image.keypoints, good_matches,
                                                  processed_data_plotter)
        end = time.time()
        print("TIME: RANSAC homography done. Elapsed time: ", end - start)

        # Filter histograms by gaussian function fitting
        start = time.time()
        histogram_filter.fit_gaussian(good_matches, good_matches_pairs, train_image.keypoints, query_image.keypoints,
                                      parameters.angle_th, parameters.length_th)
        end = time.time()
        print("TIME: Histogram filtering done. Elapsed time: ", end - start)

        fitness = histogram_filter.angle_fitness + histogram_filter.length_fitness

        if fitness > maxFit:
            # Store current configuration as best configuration
            histogram_filter.save_configuration()
            maxFit = fitness
            maxWeight = weight

    if histogram_filter.saved_configuration is not None:
        # Recover best configuration (for best weight)
        best_matches_pairs = histogram_filter.saved_configuration.filter_data_by_histogram()
        best_matches = [feature[1] for feature in best_matches_pairs]

        if parameters.plot_images:
            processed_data_plotter.plot_histogram_filtering(good_matches_pairs, best_matches_pairs, histogram_filter,
                                                            maxWeight, maxFit)

        n_final_matches = len(best_matches_pairs)

        # Initialize final displacement vectors; x and y will contain the initial points and Dx and Dy the
        # corresponding deformations
        x = np.zeros([n_final_matches, 1])
        y = np.zeros([n_final_matches, 1])
        Dx = np.zeros([n_final_matches, 1])
        Dy = np.zeros([n_final_matches, 1])

        # Proceed to calculate deformations and point maps
        for match_index, match_object in enumerate(best_matches):
            dist = [a_i - b_i for a_i, b_i in zip(
                query_image.keypoints[match_object.trainIdx].pt, train_image.keypoints[match_object.queryIdx].pt)]
            x[match_index] = int(round(train_image.keypoints[match_object.queryIdx].pt[0]))
            y[match_index] = int(round(train_image.keypoints[match_object.queryIdx].pt[1]))
            Dx[match_index] = dist[0]
            Dy[match_index] = dist[1]

        # Store new bounding box
        bb = np.array([[int(min(x)[0]), int(min(y)[0])], [int(max(x)[0]), int(max(y)[0])]])

        # If the bounding box is similar to the sample image end detection process
        if abs(int(max(y)[0]) - int(min(y)[0]) - query_image.image.shape[0])/query_image.image.shape[0] < 0.5 and \
                abs(int(max(x)[0]) - int(min(x)[0]) - query_image.image.shape[1]) / query_image.image.shape[1] < 0.5:
            break

        # Otherwise, crop down the image and repeat
        elif it+1 < parameters.crop_iterations:
            train_image.image = train_image.image[int(min(y)[0]): int(max(y)[0]), int(min(x)[0]):int(max(x)[0]), :]

            # Decrease distribution tolerance
            parameters.angle_th *= 2
            parameters.length_th *= 2

            # Track cumulative cropping
            cum_crop = np.array(cum_crop) + [int(min(x)[0]), int(min(y)[0])]

# Add cumulative cropping
bb = bb + [cum_crop, cum_crop]

# Restore shrink
bb = bb * shrink_ratio

if parameters.plot_images:
    original_data_plotter.plot_query_bounding_box(bb)
