import matplotlib.pyplot as plt
import cv2
import numpy as np

from match_geometric_filters import HistogramLogicFilter
from utils import gauss


class DataPlotter:

    def __init__(self, train_image, query_image):
        self.train_image_manager = train_image
        self.query_image_manager = query_image

    def plot_histogram_filtering(self, good_matches_pairs, best_matches_pairs, histogram_filter, weight, fitness):
        """
        Plots the result of the match histogram filtering

        :param good_matches_pairs: array of match pairs (point in image 1 vs point in image 2)
        :type good_matches_pairs: ndarray (nx2)
        :param best_matches_pairs: array of histogram-filtered match pairs (point in image 1 vs point in image 2)
        :type best_matches_pairs: ndarray (nx2)
        :param histogram_filter: histogram filtering object
        :type histogram_filter: HistogramLogicFilter
        :param weight: used weight for histogram filtering
        :type weight: float
        :param fitness: final fitness of the histogram filtering for given weight
        :type fitness: float
        """

        img1 = self.train_image_manager.image
        img2 = self.query_image_manager.image
        kp1 = self.train_image_manager.keypoints
        kp2 = self.query_image_manager.keypoints

        angle_hist = histogram_filter.angle_histogram
        length_hist = histogram_filter.length_histogram

        angle_th = histogram_filter.angle_threshold
        length_th = histogram_filter.length_threshold

        plt.figure(figsize=(20, 16))

        # Initial matches (filtered by weight)
        plt.subplot2grid((2, 4), (0, 0), colspan=3)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches_pairs, None, flags=2)
        plt.imshow(img3)
        plt.title('Unfiltered matches. Weight: {0}, Fitness: {1}'.format(weight, fitness))

        plt.subplot2grid((2, 4), (1, 0), colspan=3)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, best_matches_pairs, None, flags=2)
        plt.imshow(img3)
        plt.title('Filtered matches')

        plt.subplot(2, 4, 4)
        plt.hist(angle_hist.data, bins=angle_hist.bins, label='Test data', color='b')
        hist_fit_angle = gauss(angle_hist.bin_centres, *angle_hist.fitted_gauss_coefficients)
        plt.bar([angle_hist.fitted_gauss_coefficients[1] - angle_th * angle_hist.fitted_gauss_coefficients[2],
                 angle_hist.fitted_gauss_coefficients[1]] + angle_th / 2 * angle_hist.fitted_gauss_coefficients[2],
                np.max(angle_hist.histogram), angle_th * angle_hist.fitted_gauss_coefficients[2], alpha=0.4, color='r')
        plt.plot(angle_hist.bin_centres, hist_fit_angle, label='Fitted data', color='g')

        plt.axis([np.min(angle_hist.data), np.max(angle_hist.data), 0, np.max(angle_hist.histogram)])
        plt.title('Angle distribution')

        plt.subplot(2, 4, 8)
        plt.hist(length_hist.data, bins=length_hist.bins, label='Test data', color='b')
        hist_fit_length = gauss(length_hist.bin_centres, *length_hist.fitted_gauss_coefficients)
        plt.bar([length_hist.fitted_gauss_coefficients[1] - length_th * length_hist.fitted_gauss_coefficients[2],
                 length_hist.fitted_gauss_coefficients[1]] + length_th / 2 * length_hist.fitted_gauss_coefficients[2],
                np.max(length_hist.histogram), length_th * length_hist.fitted_gauss_coefficients[2], alpha=0.4,
                color='r')
        plt.plot(length_hist.bin_centres, hist_fit_length, label='Fitted data', color='g')

        plt.axis([np.min(length_hist.data), np.max(length_hist.data), 0, np.max(length_hist.histogram)])
        plt.title('Length distribution')
        plt.show()

    def plot_query_bounding_box(self, bounding_box):
        """
        Plots the position of query image in train image based on the found matches

        :param bounding_box: bounding box of query image in train image
        :type bounding_box: 2x2 array: row 1 = top left corner, row 2 = lower right corner
        """

        bounding_box = bounding_box.astype(int)

        plt.figure(figsize=(20, 16))
        plt.subplot2grid((1, 4), (0, 0), colspan=3)
        img3 = cv2.rectangle(self.train_image_manager.image,
                             tuple(bounding_box[1, :]), tuple(bounding_box[0, :]), 1, thickness=3)
        plt.imshow(img3)
        plt.subplot2grid((1, 4), (0, 3), colspan=1)
        plt.imshow(self.query_image_manager.image)
        plt.show()

    def plot_ransac_homography(self, matches, h_matrix, matches_mask):
        """
        Plots the result of the ransac match filtering

        :param matches: vector of original matches
        :type matches: ndarray (nx1)
        :param h_matrix: homography matrix
        :type h_matrix: array
        :param matches_mask: matches used for final ransac
        :type matches_mask: logic ndarray (nx1)
        """

        img1 = cv2.cvtColor(self.train_image_manager.image, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(self.query_image_manager.image, cv2.COLOR_BGR2GRAY)
        kp1 = self.train_image_manager.keypoints
        kp2 = self.query_image_manager.keypoints

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, h_matrix)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliners
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

        plt.imshow(img3, 'gray')
        plt.show()
