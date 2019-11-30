# numericvision

A Python 3 package for detecting numeric 
[seven-segment displays](https://en.wikipedia.org/wiki/Seven-segment_display) in images and running 
[perspective correction](https://en.wikipedia.org/wiki/Perspective_control) on them using 
[OpenCV 4](https://opencv.org).

`detect_box_sequences()` first applies a series of filters to the input image to reduce noise, bring
out primary contours and bridge gaps between digit segments (see `images.apply_filters()`). Next,
all resulting contours are analyzed by comparing their geometric properties to a set of criteria
which define what it means for a contour to resemble a seven-segment digit. Lots of assumptions are
being made here (see `config.cfg`). A contour resembling a seven-segment digit is used to
instantiate a `Box` object. Several boxes lined up in a row make up a box `Sequence`. Sequences are
contained in a `Bag`.

In other words, raw contours serve as input for an object detection pipeline which produces results
in the form of a crawlable object tree.

## Installation

```
$ brew install opencv
```
```
$ pip install numericvision
```

## Using the package

```python
from numericvision import detect_box_sequences

box_sequences = detect_box_sequences("image.jpg")
```

## Running from the command line

```
$ python -m numericvision image.jpg
```
