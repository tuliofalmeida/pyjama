
[![Build Status](https://api.travis-ci.com/python/mypy.svg?branch=master)](https://travis-ci.com/tuliofalmeida/pyjama)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![python-version](https://img.shields.io/pypi/pyversions/pyjamalib)](https://www.python.org/)
[![PyPI version fury.io](https://img.shields.io/pypi/v/pyjamalib)](https://pypi.org/project/pyjamalib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/tuliofalmeida/pyjama)

# PyJama - Python for Joint Angle Measurement Acquisition

---------------------------------------------------------
PyJama is a friendly python library for analyzing human kinematics data. Aimed at analyzing data from IMU's, MIMU's, data from optical devices and in the future tracking data from deeplearning models. The PyJAMA library was designed based on the [JAMA device](https://github.com/tuliofalmeida/jama).

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
If all went well, you should be able to execute the demo scripts under `examples/`
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
pyrat (0.x.x, /path/to/pyjamalib)
```

# Examples
-----------

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tuliofalmeida/pyjama/blob/main/PyJama_Lokomat_exemple.ipynb) Lokomat Example - Example of using the library using data extracted using JAMA.   


# Development Team:

- Tulio Almeida - [GitHub](https://github.com/tuliofalmeida) - [Scholar](https://scholar.google.com/citations?user=kkOy-JkAAAAJ&hl=pt-BR)
- Abner Cardoso - [GitHub](https://github.com/abnr) - [Scholar](https://scholar.google.com.br/citations?user=0dTid9EAAAAJ&hl=en)
- André Dantas - [GitHub](https://github.com/lordcobisco) - [Scholar](https://scholar.google.com.br/citations?user=lH6zW30AAAAJ&hl=en)

# Publications

The publications related to this project are still in the process of being published. If you publish any paper using JAMA please contact us to update [here!](mailto:tuliofalmeida@hotmail.com)

# Credits 

- [Daniele Comotti](https://github.com/danicomo/9dof-orientation-estimation) GitHub used as a basis for filters
- [Sebastian Madgwick](https://www.x-io.co.uk/res/doc/madgwick_internal_report.pdf) Reference for the manipulations of quaternions

