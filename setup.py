"""
The build/compilations setup

>> pip install -r requirements.txt
>> python setup.py install

For uploading to PyPi follow instructions
http://peterdowns.com/posts/first-time-with-pypi.html

Pre-release package
>> python setup.py sdist upload -r pypitest
>> pip install --index-url https://test.pypi.org/simple/ your-package
Release package
>> python setup.py sdist upload -r pypi
>> pip install your-package
"""
import pip
import logging
import pkg_resources
try:
    from setuptools import setup, Extension # , Command, find_packages
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup, Extension # , Command, find_packages
    from distutils.command.build_ext import build_ext


def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []


setup(
    name='mrcnn',
    version='2.1',
    url='https://github.com/matterport/Mask_RCNN',

    author='Matterport',
    author_email='',  # todo
    license='MIT',
    description='Mask R-CNN: object detection & classification & segmentation',

    packages=["mrcnn"],
    cmdclass={'build_ext': build_ext},
    install_requires=install_reqs,
    include_package_data=True,

    long_description="""This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. 
The model generates bounding boxes and segmentation masks for each instance of an object in the image. 
It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.""",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image object detection",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Image Segmentation",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
