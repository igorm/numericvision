import os.path
from setuptools import setup


HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()


setup(
    name="numericvision",
    version="0.1.0",
    description="Detects numeric displays in images using OpenCV",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/igorm/numericvision",
    author="Igor Myroshnichenko",
    author_email="imyroshnichenko@gmail.com",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    packages=["numericvision"],
    include_package_data=True,
    install_requires=["importlib_resources", "scikit-image"],
    entry_points={"console_scripts": ["igorm=numericvision.__main__:main"]},
)
