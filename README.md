# NumericVision

NumericVision is a Python 3 package for detecting numeric (seven-segment) displays in images and running perspective
correction on them using [OpenCV 4](https://opencv.org). The code is intended for demo purposes.

![Demo](images/demo.png)

## Dependencies

```
brew install opencv
```

```
pip install scikit-image
```

## Running NumericVision from the command line

```
bin/numericvision images/in/original_01.jpg
```

## Using the NumericVision package

```python
from numericvision import NumericVision

NumericVision.process('images/in/original_01.jpg')
```
