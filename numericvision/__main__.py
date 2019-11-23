"""NumericVision detects numeric displays in images using OpenCV

Usage:
------

    $ python -m numericvision image.jpg

More information:
--------

- https://github.com/igorm/numericvision
- https://pypi.org/project/numericvision
"""
from numericvision import detect_transform_dump_displays
# from .contours import Rectangle


def main(): # type: () -> None
    import sys
    args = sys.argv[1:]

    # http://nicodjimenez.github.io/boxLabel/annotate.html
    # roi_contour = Rectangle.from_tl_point_br_point((1271, 2376), (1393, 2459))

    detect_transform_dump_displays(args[0]) #, roi_contour)


if __name__ == '__main__':
    main()
