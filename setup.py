import setuptools
 
install = [
  'math',
  'numpy',
  'matplotlib.pyplot',
  'pandas',
  'sys',
  'csv'
  'os',
  'time',
  'collections',
  'scipy'
]

#python setup.py bdist_wheel 
#twine upload dist/*
 
with open("README.md", "r") as fh:
    long_description = fh.read()
  
setuptools.setup(

    name="pyjamalib",
    packages = ['pyjamalib'],
    version="0.1.7",
    author="Túlio F. Almeida",
    author_email="tuliofalmeida@hotmail.com",
    description="A library for analyze joint angles from IMU data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuliofalmeida/pyjama",
    #packages=setuptools.find_packages(),
    install_requires= ['numpy',
                      'pandas',
                      'matplotlib'],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)