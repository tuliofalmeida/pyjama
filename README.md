# PyJama - Python for Joint Angle Measurement Acquisition
---------------------------------------------------------
PyJama is a friendly python library for analyzing human kinematics data. Aimed at analyzing data from IMU's, MIMU's, data from optical devices and in the future tracking data from deeplearning models.

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
-------------------

- [Tulio Almeida]
- [Abner Cardoso] 
- [André Dantas]

<!-- Links -->
[Abner Cardoso]:https://github.com/abnr
[Tulio Almeida]:https://github.com/tuliofalmeida
[André Dantas]:https://github.com/lordcobisco

