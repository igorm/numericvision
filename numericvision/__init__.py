"""NumericVision detects numeric displays in images using OpenCV

Import `detect_displays` which returns a list of box sequences:

    >>> from numericvision import detect_displays
    >>> box_sequences = detect_displays('image.jpg')

See https://github.com/igorm/numericvision for more information
"""
import os.path
import datetime
import importlib_resources as _resources
from configparser import ConfigParser as _ConfigParser
import cv2
from .images import filter, four_point_transform
from .contours import Bag


__version__ = '0.1.0'


_cfg = _ConfigParser()
with _resources.path('numericvision', 'config.cfg') as _path:
    _cfg.read(str(_path))
URL = _cfg.get('tbd', 'test')


RGB_RED = 0, 0, 255
RGB_GREEN = 0, 255, 0
RGB_BLUE = 255, 0, 0
RGB_WHITE = 255, 255, 255
RGB_BLACK = 0, 0, 0
RGB_PURPLE = 240, 0, 159
RGB_TEAL = 255, 255, 0


def detect_displays(image_path, roi_contour=None):
    assert os.path.isfile(image_path), "File [%s] doesn't exist!" % image_path

    original_image = cv2.imread(image_path)
    filtered_image = images.filter(original_image)

    contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bag = Bag(filtered_image, contours, hierarchy[0], roi_contour)

    return bag.sequences


def detect_transform_dump_displays(image_path, roi_contour=None):
    assert os.path.isfile(image_path), "File [%s] doesn't exist!" % image_path

    out_path = "{0:%Y%m%d%H%M%S}/".format(datetime.datetime.now())
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    original_image = cv2.imread(image_path)
    cv2.imwrite(out_path + '1.jpg', original_image)
    filtered_image = images.filter(original_image)
    cv2.imwrite(out_path + '2.jpg', filtered_image)

    contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bag = Bag(filtered_image, contours, hierarchy[0], roi_contour)

    print("%i/%i" % (len(contours), bag.get_box_count()))

    contours_image = cv2.imread(image_path)
    if roi_contour:
        cv2.drawContours(contours_image, [roi_contour.points], -1, RGB_BLUE, 3)

    color = RGB_GREEN
    thickness = 1
    for sequence in bag.sequences:
        cv2.drawContours(contours_image, [sequence.get_contour().points], -1, color, thickness)
        cv2.line(contours_image, sequence.boxes[0].contour.c_point, sequence.boxes[-1].contour.c_point, color, thickness)
        cv2.putText(contours_image, "(%i,%i)" % sequence.key, sequence.boxes[-1].contour.c_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        print("%i(%i/%.2f)" % (
            sequence.key[0],
            sequence.get_box_count(),
            sequence.get_h_min_max_d_pct(),
        ))

        for box in sequence.boxes:
            # cv2.drawContours(contours_image, [box.contour.points], -1, RGB_RED, thickness)
            cv2.circle(contours_image, box.contour.c_point, 1, color, thickness)

        contour = sequence.get_contour()
        transformed_image = four_point_transform(
            original_image,
            contour.tl_point,
            contour.tr_point,
            contour.br_point,
            contour.bl_point
        )
        cv2.imwrite(out_path + "seq_%i_%i.jpg" % sequence.key, transformed_image)

    cv2.imwrite(out_path + '3.jpg', contours_image)
