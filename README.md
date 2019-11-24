# numericvision

A Python 3 package for detecting numeric (seven-segment) displays in images and running perspective correction on them
using [OpenCV 4](https://opencv.org).

![Demo](images/demo.png)

## Dependencies

```
brew install opencv
```

## Installation

```
pip install numericvision
```

## Running from the command line

```
python -m numericvision image.jpg
```

## Using the package

```python
from numericvision import detect_box_sequences

box_sequences = detect_box_sequences('image.jpg')
```
