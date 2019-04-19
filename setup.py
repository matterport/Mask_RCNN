"""
The build/compilations setup

>> conda env create -f cropmask-env.yml
>> conda activate cropmask
>> python setup.py install
"""
import pip
import logging
import pkg_resources

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

install_reqs = ["Click"]
setup(
    name="cropmask",
    version="0.0.1",
    url="https://github.com/ecohydro/CropMask_RCNN",
    author="Ryan Avery",
    author_email="ravery@ucsb.edu",
    license="MIT",
    description="Instance segmentation of agriculture in Landsat imagery",
    packages=find_packages(),
    install_requires=install_reqs,
    entry_points="""
    [console_scripts]
    order_landsat_to_azure=cropmask.download.order_landsat_to_azure:cli
    """,
    include_package_data=True,
    python_requires=">=3.6",
    long_description="""Contains modules for downloading, preprocessing and training mrcnn on Landsat satellite imagery in order to detect center pivot agriculture and other land cover types. Runs on Microsoft Azure.""",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Image Segmentation",
        "Topic :: Scientific/Engineering :: Remote Sensing",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="image instance segmentation object detection r-cnn tensorflow keras landsat agriculture",
)
