"""numericvision detects numeric displays in images using OpenCV

Import `detect_box_sequences` which returns sequences of seven-segment digit boxes:

    >>> from numericvision import detect_box_sequences
    >>> box_sequences = detect_box_sequences("image.jpg")

See https://github.com/igorm/numericvision for more information
"""
import os.path
import datetime
from configparser import ConfigParser
import importlib_resources
import cv2
from .images import apply_filters, four_point_transform
from .contours import Bag


__version__ = "0.1.0"


_config = ConfigParser()
with importlib_resources.path("numericvision", "config.cfg") as _config_path:
    _config.read(str(_config_path))

BOX_MIN_IMAGE_AREA_PCT = float(_config.get("box", "min_image_area_pct"))
BOX_MIN_ASPECT_RATIO = float(_config.get("box", "min_aspect_ratio"))
BOX_MAX_ASPECT_RATIO = float(_config.get("box", "max_aspect_ratio"))
BOX_MAX_TALL_NARROW_ASPECT_RATIO = float(_config.get("box", "max_tall_narrow_aspect_ratio"))
BOX_DUPLICATE_MAX_X_D = int(_config.get("box", "duplicate_max_x_d"))
BOX_DUPLICATE_MAX_Y_D = int(_config.get("box", "duplicate_max_y_d"))
BOX_DUPLICATE_MAX_W_D = int(_config.get("box", "duplicate_max_w_d"))
BOX_DUPLICATE_MAX_H_D = int(_config.get("box", "duplicate_max_h_d"))
BOX_SHARD_MAX_X_D = int(_config.get("box", "shard_max_x_d"))
BOX_SHARD_MIN_Y_D = int(_config.get("box", "shard_min_y_d"))
BOX_SHARD_MAX_H_Y_D = int(_config.get("box", "shard_max_h_y_d"))
BOX_SHARD_MAX_W_D = int(_config.get("box", "shard_max_w_d"))
BOX_SHARD_MAX_H_D = int(_config.get("box", "shard_max_h_d"))
SEQ_MIN_BOX_COUNT = int(_config.get("sequence", "min_box_count"))
SEQ_MAX_AVG_DISTANCE_TO_CENTER_LINE = int(
    _config.get("sequence", "max_avg_distance_to_center_line")
)
SEQ_MAX_X_D_MIN_MAX_D_PCT = int(_config.get("sequence", "max_x_d_min_max_d_pct"))
SEQ_MAX_H_MIN_MAX_D_PCT = int(_config.get("sequence", "max_h_min_max_d_pct"))

RGB_RED = 0, 0, 255
RGB_GREEN = 0, 255, 0
RGB_BLUE = 255, 0, 0
RGB_WHITE = 255, 255, 255
RGB_BLACK = 0, 0, 0
RGB_PURPLE = 240, 0, 159
RGB_TEAL = 255, 255, 0


def detect_box_sequences(image_path, roi_contour=None):
    """Analyzes image contours and detects sequences of seven-segment digit boxes."""
    assert os.path.isfile(image_path), "File [%s] doesn't exist!" % image_path

    original_image = cv2.imread(image_path)
    filtered_image = apply_filters(original_image)

    contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bag = Bag(filtered_image, contours, hierarchy[0], roi_contour)

    return bag.sequences


def detect_transform_dump_box_sequences(image_path, roi_contour=None):
    """Analyzes image contours, detects sequences of seven-segment digit boxes, runs perspective
    correction on detected numeric displays, writes work files.
    """
    assert os.path.isfile(image_path), "File [%s] doesn't exist!" % image_path

    out_path = "{0:%Y%m%d%H%M%S}/".format(datetime.datetime.now())
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    original_image = cv2.imread(image_path)
    cv2.imwrite(out_path + "1.jpg", original_image)
    filtered_image = apply_filters(original_image)
    cv2.imwrite(out_path + "2.jpg", filtered_image)

    contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bag = Bag(filtered_image, contours, hierarchy[0], roi_contour)

    print("%i/%i/%i" % (len(contours), bag.get_box_count(), len(bag.sequences)))

    contours_image = cv2.imread(image_path)
    if roi_contour:
        cv2.drawContours(contours_image, [roi_contour.points], -1, RGB_BLUE, 3)

    color = RGB_GREEN
    thickness = 1
    for sequence in bag.sequences:
        cv2.drawContours(contours_image, [sequence.get_contour().points], -1, color, thickness)
        cv2.line(
            contours_image,
            sequence.boxes[0].contour.c_point,
            sequence.boxes[-1].contour.c_point,
            color,
            thickness,
        )
        cv2.putText(
            contours_image,
            "(%i, %i)" % sequence.key,
            sequence.boxes[-1].contour.c_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )
        print(
            "(%i, %i) %i/%.2f"
            % (*sequence.key, sequence.get_box_count(), sequence.get_h_min_max_d_pct(),)
        )

        for box in sequence.boxes:
            cv2.drawContours(contours_image, [box.contour.points], -1, RGB_RED, thickness)
            cv2.circle(contours_image, box.contour.c_point, 1, color, thickness)

        contour = sequence.get_contour()
        transformed_image = four_point_transform(
            original_image, contour.tl_point, contour.tr_point, contour.br_point, contour.bl_point
        )
        cv2.imwrite(out_path + "seq_%i_%i.jpg" % sequence.key, transformed_image)

    cv2.imwrite(out_path + "3.jpg", contours_image)
