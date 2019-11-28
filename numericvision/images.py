"""Functions for manipulating and applying filters to images."""
import numpy as np
import cv2
from skimage import exposure
import numericvision as nv


def apply_filters(original_image):
    """Applies filters to an image to reduce noise and expose primary contours."""
    # Initial processing
    filtered_image = cv2.bilateralFilter(original_image, 11, 17, 17)
    filtered_image = cv2.GaussianBlur(filtered_image, (3, 3), 0)
    filtered_image = cv2.erode(filtered_image, np.ones((5, 5), np.uint8), iterations=1)

    # filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    # filtered_image = cv2.threshold(filtered_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    edged_image = detect_edges(filtered_image)

    contours_image = np.zeros(original_image.shape, np.uint8)
    contours, hierarchy = cv2.findContours(edged_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for points, node in zip(contours, hierarchy[0]):
        cv2.drawContours(contours_image, [points], -1, nv.RGB_WHITE, 1)

    # Secondary processing
    filtered_image = cv2.morphologyEx(contours_image, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    filtered_image = exposure.rescale_intensity(filtered_image, out_range=(0, 255))

    return filtered_image


def detect_edges(image, sigma=0.33):
    """Detects edges in an image."""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv2.Canny(image, lower, upper)


def crop(image, contour):
    """Crops the image by the provided contour."""
    tl_x, tl_y = contour.tl_point
    br_x, br_y = contour.br_point

    return image[int(tl_y) : int(br_y), int(tl_x) : int(br_x)]


def four_point_transform(image, tl, tr, br, bl):
    """Performs perspective correction on an area of the provided image.
    http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example
    """
    rect = np.array([tl, tr, br, bl], dtype="float32")

    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped_image
