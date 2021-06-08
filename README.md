
[![Build Status](https://api.travis-ci.com/python/mypy.svg?branch=master)](https://travis-ci.com/tuliofalmeida/pyjama)
[![python-version](https://upload.wikimedia.org/wikipedia/commons/a/a5/Blue_Python_3.8_Shield_Badge.svg)](https://www.python.org/)
[![PyPI version fury.io](https://img.shields.io/pypi/v/pyjamalib)](https://pypi.org/project/pyjamalib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![last-commit](https://img.shields.io/github/last-commit/tuliofalmeida/pyjama)](https://github.com/tuliofalmeida/pyjama/commits/main)
[![downloads](https://img.shields.io/pypi/dm/pyjamalib)](https://pypi.org/project/pyjamalib/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/tuliofalmeida/pyjama)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tuliofalmeida/pyjama/blob/main/PyJama_JAMA_exemple.ipynb)
[![stars](https://img.shields.io/github/stars/tuliofalmeida?style=social)](https://github.com/tuliofalmeida/pyjama/stargazers)

# PyJama - Python for Joint Angle Measurement and Acquisition
---------------------------------------------------------

PyJama is open access project that was developed during my master's work at [Edmond and Lily Safra International Institute of Neuroscience](https://github.com/isd-iin-els) of [Santos Dumont Insitute](http://www.institutosantosdumont.org.br/unidade/instituto-neurociencias-iinels/). PyJama is a user friendly python library for analyzing human kinematics data. Aimed at analyzing data from IMU's, MIMU's, data from optical devices and in the future tracking data from deeplearning models. The PyJama library was designed based on the [JAMA device](https://github.com/tuliofalmeida/jama).


# Contents
-----------
- [Installation](#Installation)
- [Examples](#Examples)
- [Contributing](#contributing)
- [Development Team](#Development-Team)  
- [Publications](#Publications)
- [Credits](#Credits)   

# Installation
--------------

The latest stable release is available on PyPI, and you can install it by saying
```
pip install pyjamalib
```
Anaconda users can install using ``conda-forge``:
```
conda install -c conda-forge pyjamalib
```

To build PyJama from source, say `python setup.py build`.
Then, to install PyJama, say `python setup.py install`.
If all went well, you should be able to execute the demo scripts under [examples](#Examples)
(OS X users should follow the installation guide given below).

Alternatively, you can download or clone the repository and use `pip` to handle dependencies:

```
unzip pyjamalib.zip
pip install -e pyjamalib
```
or
```
git clone https://github.com/tuliofalmeida/pyjama
pip install -e pyjamalib
```

By calling `pip list` you should see `pyjamalib` now as an installed package:
```
pyjamalib (0.x.x, /path/to/pyjamalib)
```

# Examples
-----------

- Example of using the library using data extracted using JAMA. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/tuliofalmeida/pyjama/blob/main/PyJama_JAMA_exemple.ipynb)      
- Example of using the library using data extracted using Vicon and Xsens. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tuliofalmeida/pyjama/blob/main/Pyjama_total_capture_example.ipynb) 

# Contributing
--------------

For minor fixes of code and documentation, please go ahead and submit a pull request.  A gentle introduction to the process can be found [here](https://www.freecodecamp.org/news/a-simple-git-guide-and-cheat-sheet-for-open-source-contributors/).

Check out the list of issues that are easy to fix. Working on them is a great way to move the project forward.

Larger changes (rewriting parts of existing code from scratch, adding new functions to the core, adding new libraries) should generally be discussed by opening an issue first. PRs with such changes require testing and approval.

Feature branches with lots of small commits (especially titled "oops", "fix typo", "forgot to add file", etc.) should be squashed before opening a pull request. At the same time, please refrain from putting multiple unrelated changes into a single pull request.

# Development Team:
-------------------

- Tulio Almeida - [GitHub](https://github.com/tuliofalmeida) - [Google Scholar](https://scholar.google.com/citations?user=kkOy-JkAAAAJ&hl=en)
- Abner Cardoso - [GitHub](https://github.com/abnr) - [Google Scholar](https://scholar.google.com.br/citations?user=0dTid9EAAAAJ&hl=en)
- André Dantas - [GitHub](https://github.com/lordcobisco) - [Google Scholar](https://scholar.google.com.br/citations?user=lH6zW30AAAAJ&hl=en)

# Publications
--------------

The publications related to this project are still in the process of being published. If you publish any paper using JAMA please contact us to update [here!](mailto:tuliofalmeida@hotmail.com)

# Credits 
---------

- [Daniele Comotti](https://github.com/danicomo/9dof-orientation-estimation) GitHub used as a basis for filters
- [Sebastian Madgwick](https://www.x-io.co.uk/res/doc/madgwick_internal_report.pdf) Reference for the manipulations of quaternions
- [Center for Vision, Speech & Signal Processing - University of Surrey](https://www.surrey.ac.uk/centre-vision-speech-signal-processing). For making available the [Total Capture dataset](https://cvssp.org/data/totalcapture/) used to develop the library example. Reference paper: [Trumble et. al., 2017](https://core.ac.uk/download/pdf/84589062.pdf)

